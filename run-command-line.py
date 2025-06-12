
"""
Batch knowledge‑graph generator for transcript JSON files.

Usage
-----
python batch_kg.py \
   --input-dir "C:\\Users\\ryanw\\PycharmProjects\\kg-gen\\MINE\\input_stories" \
   --output-dir "C:\\Users\\ryanw\\PycharmProjects\\kg-gen\\MINE\\output_graphs" \
   --config "config.yaml" \
   --llm-model "gpt-4o-mini" \
   --embedding-model "text-embedding-3-small"
"""
import os
import json
import csv
import argparse
import logging
import bisect
import re
import yaml
from typing import List, Dict

from autokg import autoKG
import utils

# ---------- logging ----------
logging.basicConfig(
    filename="kg_batch.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ---------- text processing (mirrors KnowledgeGraphTab) ----------
MAX_WORDS = 100


def _split_block(spk: str, text: str) -> List[Dict[str, str]]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [{"speaker": spk, "text": text.strip()}]

    sentences = re.split(r"(?<=[\.?!])\s+", text)
    if len(sentences) > 1:
        counts = [len(s.split()) for s in sentences]
        prefix = [0]
        for c in counts:
            prefix.append(prefix[-1] + c)
        half = prefix[-1] / 2
        i0 = bisect.bisect_left(prefix, half)
        best_i = min(max(i0, 1), len(sentences) - 1)
        for cand in (i0 - 1, i0 + 1):
            if 1 <= cand < len(prefix) and abs(half - prefix[cand]) < abs(
                half - prefix[best_i]
            ):
                best_i = cand
        left = " ".join(sentences[:best_i])
        right = " ".join(sentences[best_i:])
        return _split_block(spk, left) + _split_block(spk, right)

    mid = len(words) // 2
    left = " ".join(words[:mid])
    right = " ".join(words[mid:])
    return _split_block(spk, left) + _split_block(spk, right)


def process_entries(entries: List[Dict]) -> List[Dict]:
    merged: List[List[str]] = []
    for seg in entries:
        spk = seg.get("speaker", "")
        txt = seg.get("text", "").strip()
        if not merged or merged[-1][0] != spk:
            merged.append([spk, txt])
        else:
            merged[-1][1] += " " + txt
    out: List[Dict] = []
    for spk, text in merged:
        out.extend(_split_block(spk, text))
    return out


# ---------- helpers ----------
def load_transcript(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        tr = []
        for seg in data["segments"]:
            tr.append(
                {
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "speaker": seg.get("speaker", "speaker_0"),
                    "text": seg.get("text", "").strip(),
                }
            )
        return tr
    return data  # old format list[dict]


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def export_csvs(nodes: List[str], triples: List[Dict], out_dir: str):
    ensure_dir(out_dir)
    nodes_file = os.path.join(out_dir, "nodes.csv")
    edges_file = os.path.join(out_dir, "edges.csv")

    connected = set()
    for t in triples:
        connected.add(t["subject"])
        obj = t["object"]
        if isinstance(obj, list):
            connected.update(obj)
        else:
            connected.add(obj)

    with open(nodes_file, "w", newline="", encoding="utf-8") as nf:
        writer = csv.writer(nf)
        writer.writerow(["id", "label"])
        for n in sorted(nodes):
            if n in connected:
                writer.writerow([n, n])

    with open(edges_file, "w", newline="", encoding="utf-8") as ef:
        writer = csv.writer(ef)
        writer.writerow(["source", "relation", "target"])
        for t in triples:
            subj, rel, obj = t["subject"], t["relation"], t["object"]
            if isinstance(obj, list):
                for o in obj:
                    writer.writerow([subj, rel, o])
            else:
                writer.writerow([subj, rel, obj])


def build_graph(
    texts: List[str],
    sources: List[str],
    llm_model: str,
    embedding_model: str,
    cfg: dict,
):
    # -------- API key handling (mirrors GUI) --------
    def fetch_keys(section: str, num: int) -> List[str]:
        secs = cfg.get(section, {})
        if num == 1:
            return [secs.get("api_key", "").strip()]
        if num == 3:
            return [
                secs.get("AWS_ACCESS_KEY_ID", "").strip(),
                secs.get("AWS_SECRET_ACCESS_KEY", "").strip(),
                secs.get("AWS_REGION_NAME", "").strip(),
            ]
        return [""] * num

    llm_parent = utils.determine_llm_parent(llm_model)
    emb_parent = utils.determine_embedding_parent(embedding_model)

    if llm_parent == "bedrock_llm":
        llm_keys = fetch_keys("bedrock", 3)
    elif llm_parent == "ollama_llm":
        llm_keys = [cfg.get("ollama", {}).get("api_base", "").strip(), "", ""]
    else:
        prov = llm_parent.replace("_llm", "")
        llm_keys = fetch_keys(prov, 1) + ["", ""]

    if emb_parent == "bedrock_embedding":
        emb_keys = fetch_keys("bedrock", 3)
    elif emb_parent == "local_embedding":
        emb_keys = ["", "", ""]
    else:
        prov = emb_parent.replace("_embedding", "")
        emb_keys = fetch_keys(f"{prov}_embedding", 1) + ["", ""]

    # Set environment variables so autoKG can pick them up if needed
    utils.set_env_variables(llm_model, embedding_model, llm_keys, emb_keys)

    auto_kg = autoKG(
        texts=texts,
        source=sources,
        embedding_model=embedding_model,
        llm_model=llm_model,
        embedding_api_key=emb_keys[0],
        llm_api_key=llm_keys[0],
        main_topic="",
        embed=True,
        embedding_key2=emb_keys[1],
        embedding_key3=emb_keys[2],
        llm_key2=llm_keys[1],
        llm_key3=llm_keys[2],
    )

    # Same pipeline as GUI
    auto_kg.make_graph(k=5, method="annoy", similarity="angular", kernel="gaussian")
    auto_kg.remove_same_text(use_nn=True, n_neighbors=3, thresh=1e-6, update=True)
    auto_kg.cluster(
        n_clusters=None,
        clustering_method="k_means",
        max_texts=8,
        select_mtd="similarity",
        prompt_language="English",
        num_topics=30,
        max_length=3,
        post_process=True,
        add_keywords=True,
        verbose=False,
    )
    auto_kg.coretexts_seg_individual(
        trust_num=5,
        core_labels=None,
        dist_metric="cosine",
        negative_multiplier=3,
        seg_mtd="laplace",
        return_mat=True,
        connect_threshold=1,
    )
    threshold = auto_kg.apply_dynamic_threshold(50)
    print(f"Applied median threshold: {threshold:.4f}")
    edges = auto_kg.build_entity_relationships(transcript_str=" ".join(texts))
    filtered_edges = [(s, r, o, d) for s, r, o, d in edges if r]

    nodes = auto_kg.keywords.copy()
    triples = []
    for kw1, relation, kw2, direction in filtered_edges:
        if direction == "forward":
            triples.append({"subject": kw1, "relation": relation, "object": kw2})
        else:
            triples.append({"subject": kw2, "relation": relation, "object": kw1})
    return nodes, triples


# ---------- main ----------
def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_files = sorted(
        f
        for f in os.listdir(args.input_dir)
        if f.endswith(".json") and f not in {"001.json", "002.json"}
    )
    if not input_files:
        logging.warning("No input JSON files found.")
        return

    total = len(input_files)
    logging.info("Starting batch generation for %d transcripts", total)

    for idx, fname in enumerate(input_files, 1):
        base = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.output_dir, base)
        if os.path.exists(os.path.join(out_dir, "nodes.csv")):
            logging.info("(%d/%d) %s already processed, skipping", idx, total, fname)
            continue

        try:
            logging.info("(%d/%d) Processing %s", idx, total, fname)

            transcript = load_transcript(os.path.join(args.input_dir, fname))
            processed = process_entries(transcript)
            texts = [e["text"] for e in processed]
            sources = [e["speaker"] for e in processed]

            nodes, triples = build_graph(
                texts, sources, args.llm_model, args.embedding_model, cfg
            )
            export_csvs(nodes, triples, out_dir)
            logging.info("Finished %s (%d nodes, %d triples)", fname, len(nodes), len(triples))
        except Exception as e:
            logging.exception("Error processing %s: %s", fname, e)

    logging.info("Batch run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch knowledge‑graph generator")
    parser.add_argument("--input-dir", required=True, help="Folder containing transcript JSON files")
    parser.add_argument("--output-dir", required=True, help="Folder where graph folders will be created")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml with API keys")
    parser.add_argument("--llm-model", required=True, help="LLM model name (must match GUI drop‑down)")
    parser.add_argument("--embedding-model", required=True, help="Embedding model name")
    args = parser.parse_args()
    main(args)
