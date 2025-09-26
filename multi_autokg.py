############ multi_autokg.py ############
"""
MultiSpeakerAutoKG – an extension of autoKG that is aware of speakers and.  
It re‑uses almost everything from the original autoKG but changes three 
things:
  • accepts the *full* transcript (each block must have a `speaker` key)
  • formats chunks with explicit speaker labels (e.g. "S1: text …")
  • extends the relation‑extraction prompt so the LLM returns `speaker` in 
    one shot – no extra calls are required.

Only the parts that differ from autoKG are included below; everything else is
inherited unchanged.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import json
from autokg import autoKG, extract_json  # type: ignore
from utils import get_num_tokens, process_strings, remove_duplicates  # type: ignore
from litellm import get_max_tokens
import re

EXTRACT_RELATIONSHIP_BATCH_SIZE = 20

class MultiSpeakerAutoKG(autoKG):
    """Speaker‑aware variant of *autoKG*.

    Parameters
    ----------
    transcript : list[dict]
        Each dict **must** have at least the keys ``speaker`` and ``text``.
        Additional keys (start/end, etc.) are ignored here but preserved on
        ``self.transcript_blocks`` for downstream use.
    *args, **kwargs
        Forwarded to :class:`autokg.autoKG` – the parent class still wants the
        *texts* and *source* lists, which we derive from ``transcript``.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        full_transcript: List[Dict[str,str]],
        embedding_model: str,
        llm_model: str,
        embedding_api_key: str,
        llm_api_key: str,
        main_topic: str = "",
        embed: bool = True,
        embedding_key2: str = "",
        embedding_key3: str = "",
        llm_key2: str = "",
        llm_key3: str = "",
    ):
        texts  = [seg["text"]    for seg in full_transcript]
        source = [seg["speaker"] for seg in full_transcript]

        super().__init__(
            texts=texts,
            source=source,
            embedding_model=embedding_model,
            llm_model=llm_model,
            embedding_api_key=embedding_api_key,
            llm_api_key=llm_api_key,
            main_topic=main_topic,
            embed=embed,
            embedding_key2=embedding_key2,
            embedding_key3=embedding_key3,
            llm_key2=llm_key2,
            llm_key3=llm_key3,
        )
        self.transcript = self._format_transcript(full_transcript)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_transcript(self, full_transcript) -> str:
        """Return the entire transcript as a single string with speaker tags."""
        parts = [f"{blk['speaker']}: {blk['text']}" for blk in full_transcript]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Relation extraction – speaker aware
    # ------------------------------------------------------------------

    def batch_extract_relationships_for_chunk(self,
                                              chunk_text: str,
                                              pairs: List[Tuple[str, str]]):
        """Override: ask the LLM for **speaker** too."""
        print(">>> [MultiSpeakerAutoKG] batch_extract_relationships_for_chunk")
        print(f"    incoming chunk length: {len(chunk_text)} chars")
        print(f"    incoming pairs count: {len(pairs)}")

        instructions = (
            """You are an expert discourse analyst. Follow these rules **exactly**:\n
            1. The context is a transcript where each utterance is prefixed by “SpeakerName:”.\n
               • Capture that prefix as the `speaker` for any relation found in its text.\n
            2. You will be given a list of entity pairs. For each pair:\n
               a. Decide which mention is SUBJECT (actor) and which is OBJECT (receiver).\n
               b. If there is no clear direction, set `direction` to `none`.\n
            3. Propose **up to 3** distinct predicates (relations) between the subject and object.\n
                a. These relations need to be specific and meaningful, do not use ones like "relation", "relate", or "affect"
            5. Supply a brief `rationale` (≤15 words) explaining why this relation holds.\n
            6. **Output only** a single JSON object, with no extra keys or commentary:\n

            Use this example to show you what to do:
            content:
            Alice: Single-use plastics fill landfills and pollute waterways.  
            Bob: Reusable bags cost only a few cents more, and many stores already offer them.  
            Alice: Small businesses cannot afford to stock alternatives at scale.  
            Bob: That may be true, but some cities with bans see 40 percent less plastic litter.  
            Alice: That statistic comes from a report funded by an anti-plastic NGO.  
            Alice: The report was published in 2021 by the EPA.
            Bob: Our policies and infrastructure should evolve.

            triple pairs:
            1. Single-use plastics, landfills  
            2. Single-use plastics, waterways  
            3. Reusable bags, money  
            4. Stores, Reusable bags  
            5. alternatives at scale, small businesses  
            6. Cities with bans, plastic litter  
            7. report, anti-plastic NGO  
            8. policies, infrastructure
            9. the report, EPA  

            {
            results: [
                {"pair_index": 1, "direction": "forward", "relationship": "fill",  "speaker": "Alice", "rationale": "Alice says plastics fill landfills."},
                {"pair_index": 2, "direction": "forward", "relationship": "pollute",  "speaker": "Alice", "rationale": "Alice states plastics pollute waterways."},
                {"pair_index": 3, "direction": "forward", "relationship": "cost a little more",  "speaker": "Bob", "rationale": "Bob claims reusable bags cost slightly more."},
                {"pair_index": 4, "direction": "forward", "relationship": "offer",  "speaker": "Bob", "rationale": "Bob says many stores already offer them."},
                {"pair_index": 5, "direction": "reverse", "relationship": "cannot afford to stock",  "speaker": "Alice", "rationale": "Alice says small businesses cannot afford large-scale alternatives."},
                {"pair_index": 6, "direction": "forward", "relationship": "report less",  "speaker": "Bob", "rationale": "Bob reports 40% less plastic litter in such cities."},
                {"pair_index": 7, "direction": "forward", "relationship": "funded by",  "speaker": "Alice", "rationale": "Alice says the NGO funded the report."},
                {"pair_index": 8, "direction": "", "relationship": "",  "speaker": "Bob", "rationale": "Bob doesn't indicate a relation between the two"},
                {"pair_index": 9, "direction": "forward", "relationship": "was published by", "speaker": "Alice", "rationale": "Alice says the EPA published the report."}
            ]}"""
        )

        # index the pairs
        pairs_with_index = [(i, a, b)
                            for i, (a, b) in enumerate(pairs, start=1)]
        print(f"    pairs_with_index sample: {pairs_with_index[:5]} …")

        base_prompt = (
            instructions
            + "\nContext:\n\"\"\"\n" + chunk_text + "\n\"\"\"\n\n"
            "Pairs to process:\n"
        )
        print("    base_prompt head:")
        print(base_prompt.splitlines()[:3])

        def _pairs_text(lst):
            return "".join(f"{i}. {a} -- {b}\n" for i, a, b in lst)

        max_tokens = get_max_tokens(model=self.llm_model)
        print(f"    max_tokens for LLM: {max_tokens}")

        def _batches(lst):
            print(f"    → _batches called on {len(lst)} pairs")
            if len(lst) > EXTRACT_RELATIONSHIP_BATCH_SIZE:
                out = []
                for i in range(0, len(lst), EXTRACT_RELATIONSHIP_BATCH_SIZE):
                    sub = lst[i:i+EXTRACT_RELATIONSHIP_BATCH_SIZE]
                    out.extend(_batches(sub))
                return out
            prompt_len = self.encoding(model=self.llm_model,
                                       text=base_prompt + _pairs_text(lst))
            print(f"       prompt token count: {prompt_len}")
            if prompt_len <= max_tokens or len(lst) == 1:
                print(f"       → one batch of size {len(lst)}")
                return [lst]
            mid = len(lst) // 2
            return _batches(lst[:mid]) + _batches(lst[mid:])

        batches = _batches(pairs_with_index)
        print(f"    total batches: {len(batches)}")

        combined = []
        for bi, batch in enumerate(batches, start=1):
            print(f"    --- Batch #{bi} with {len(batch)} pairs ---")
            prompt = base_prompt + _pairs_text(batch)
            resp, *_ = self._call_llm(prompt)
            print(f"      Raw LLM response (head): {resp!r}…")
            
            try:
                raw = extract_json(resp)
                print(raw)

                # 1) strip odd leading/trailing ellipses
                raw = re.sub(r"^\s*…+|\…+\s*$", "", raw)

                # 2) drop any trailing commas before } or ]
                raw = re.sub(r",\s*}", "}", raw)
                raw = re.sub(r",\s*]", "]", raw)
                data = json.loads(raw)
                print(f"      Parsed JSON top‐level keys: {list(data.keys())}")
                if 'results' in data:
                    print(f"    -> got {len(data['results'])} result objects")
                    if data['results']:
                        print(f"    -> result[0] fields: {list(data['results'][0].keys())}")
            except Exception as e:
                print(f"      JSON parse error: {e}")
                data = {}
            if isinstance(data, dict):
                combined.extend(data.get("results", []))
            elif isinstance(data, list):
                combined.extend(data)
        print(f"<<< [MultiSpeakerAutoKG] batch_extract done: collected {len(combined)} results")
        return combined


    # def build_entity_relationships(self,
    #                                fallback_if_no_chunk=True):
    #     """Same algorithm but returns 6-tuples with speaker."""
    #     print(">>> [MultiSpeakerAutoKG] build_entity_relationships")
    #     base_edges = super().build_entity_relationships(
    #         fallback_if_no_chunk=fallback_if_no_chunk)
    #     print(f"    super().build_entity_relationships returned {len(base_edges)} edges")

    #     enriched = []
    #     for sub, rel, obj, direction in base_edges:
    #         key1, key2 = (sub, obj), (obj, sub)
    #         match = next(
    #             (
    #                 r for r in getattr(self, "_last_combined_results", [])
    #                 if (r.get("subject_entity"), r.get("object_entity")) in (key1, key2)
    #                 and r.get("relationship") == rel
    #             ),
    #             None
    #         )
    #         speaker = match.get("speaker", "") if match else "<none>"
    #         enriched.append((sub, rel, obj, direction, speaker))
    #     print(f"<<< [MultiSpeakerAutoKG] build_entity_relationships enriched to {len(enriched)} tuples")
    #     return enriched
    
    # ------------------------------------------------------------------
    # Internal – single helper to keep parent’s get_completion pattern.
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str):
        """Thin wrapper so we only import *get_completion* once here."""
        from utils import get_completion  # local import to avoid cycles
        return get_completion(prompt,
                              model_name=self.llm_model,
                              temperature=self.temperature,
                              top_p=self.top_p,
                              llm_api_key=self.llm_api_key)


