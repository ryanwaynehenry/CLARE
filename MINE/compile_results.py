"""
overlap_histograms.py

Find overlapping article IDs across FOUR results directories (filenames like
"<id>_<anything-containing-'results'>.json"), extract accuracies, and compute
histograms and summary stats restricted to the intersection.

Assumptions:
  - JSON files are lists whose *last* entry has an "accuracy" field, typically like "53.33%".
  - Leading zeros in filenames are fine; IDs are parsed as decimal integers from the prefix.

Example:
  python overlap_histograms.py path/to/CLARE path/to/KGGen path/to/GraphRAG path/to/OpenIE \
    --labels CLARE KGGen GraphRAG OpenIE --out-csv overlap_scores.csv

Outputs:
  - Prints histogram bin edges and per-system counts (percent scale 0..100)
  - Prints summary stats (n, mean, sd) for overlapping IDs only
  - Optionally writes a per-article CSV aligned across systems
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FILENAME_REGEX = re.compile(r'^(\d+)_.*results.*\.json$', flags=re.IGNORECASE)
PERCENT_RE = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*%?\s*$')

def parse_args():
    p = argparse.ArgumentParser(description="Overlap-only histogram/stats across four results directories.")
    p.add_argument("dir1", type=Path, help="Results dir 1 (e.g., CLARE)")
    p.add_argument("dir2", type=Path, help="Results dir 2 (e.g., KGGen)")
    p.add_argument("dir3", type=Path, help="Results dir 3 (e.g., GraphRAG)")
    p.add_argument("dir4", type=Path, help="Results dir 4 (e.g., OpenIE)")
    p.add_argument("--labels", nargs=4, default=["Dir1", "Dir2", "Dir3", "Dir4"],
                   help="Labels for the four systems (in the same order).")
    p.add_argument("--bins", type=int, default=16, help="Number of equal-width bins in [0,100]. Default: 16.")
    p.add_argument("--pattern", default=r'^(\d+)_.*results.*\.json$',
                   help="Regex to match filenames (group 1 must capture numeric ID).")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional CSV path with per-article overlapping accuracies.")
    p.add_argument("--verbose", action="store_true", help="Print notes about skipped files, parsing, etc.")
    return p.parse_args()

def id_to_file_map(directory: Path, pattern: str, verbose: bool=False) -> Dict[int, Path]:
    """Return {id -> file} for files matching the pattern. If duplicate IDs, keep lexicographically last."""
    regex = re.compile(pattern, flags=re.IGNORECASE)
    mapping: Dict[int, Path] = {}
    for name in os.listdir(directory):
        m = regex.match(name)
        if not m:
            continue
        try:
            art_id = int(m.group(1))
        except ValueError:
            if verbose:
                print(f"[skip] cannot parse id in {name}", file=sys.stderr)
            continue
        path = directory / name
        # Keep the lexicographically last file if duplicates exist
        prev = mapping.get(art_id)
        if prev is None or str(path.name) > str(prev.name):
            mapping[art_id] = path
    return mapping

def parse_percent(s: str) -> Optional[float]:
    """
    Accepts "53.33%" -> 53.33, "53.33" -> 53.33, "0.8" -> 0.8 (treated as percent already).
    We cannot infer scale perfectly; for consistency with your histograms (0..100),
    we aim to return values on a 0..100 scale whenever possible.
    """
    m = PERCENT_RE.match(s)
    if not m:
        return None
    val = float(m.group(1))
    # If a % sign is present, it's clearly a percent 0..100 already.
    if "%" in s:
        return val
    # Heuristic: if value <= 1, treat as fraction and convert to percent.
    return val * 100.0 if 0.0 <= val <= 1.0 else val

def load_accuracy_percent(path: Path, verbose: bool=False) -> Optional[float]:
    """
    Load accuracy from JSON on a 0..100 scale.
    Expected format: list of dicts, with the last dict containing "accuracy",
    typically a string like "53.33%".
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        if verbose:
            print(f"[skip] failed to load {path}: {e}", file=sys.stderr)
        return None

    # Typical format: list with last entry carrying 'accuracy'
    if isinstance(data, list) and data:
        last = data[-1]
        if isinstance(last, dict) and "accuracy" in last:
            acc = last["accuracy"]
            if isinstance(acc, (int, float)):
                # Return as-is if looks like 0..100; else scale if it's a plausible fraction.
                return float(acc) * 100.0 if 0.0 <= float(acc) <= 1.0 else float(acc)
            if isinstance(acc, str):
                val = parse_percent(acc)
                if val is not None:
                    return val
        if verbose:
            print(f"[warn] no usable 'accuracy' in last item for {path.name}", file=sys.stderr)
        return None

    # Fallback: dict with "accuracy"
    if isinstance(data, dict) and "accuracy" in data:
        acc = data["accuracy"]
        if isinstance(acc, (int, float)):
            return float(acc) * 100.0 if 0.0 <= float(acc) <= 1.0 else float(acc)
        if isinstance(acc, str):
            val = parse_percent(acc)
            if val is not None:
                return val

    if verbose:
        print(f"[skip] unsupported JSON shape or missing 'accuracy' in {path.name}", file=sys.stderr)
    return None

def mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)

def main():
    args = parse_args()
    dirs = [args.dir1, args.dir2, args.dir3, args.dir4]
    labels = args.labels
    if len(labels) != 4:
        print("Error: --labels must provide exactly four items (one per directory).", file=sys.stderr)
        sys.exit(2)

    # Validate dirs
    for d in dirs:
        if not d.exists() or not d.is_dir():
            print(f"Error: {d} is not a directory", file=sys.stderr)
            sys.exit(2)

    # Build id->file maps
    maps = [id_to_file_map(d, args.pattern, verbose=args.verbose) for d in dirs]
    id_sets = [set(m.keys()) for m in maps]
    overlap_ids = sorted(set.intersection(*id_sets))

    print("=== Overlap Summary ===")
    for lbl, m in zip(labels, maps):
        print(f"{lbl}: files matched = {len(m)}")
    print(f"Overlapping article IDs across ALL four dirs: n = {len(overlap_ids)}")

    if len(overlap_ids) == 0:
        print("No overlapping IDs across all four directories.", file=sys.stderr)
        sys.exit(1)

    # Load accuracies (0..100) aligned by overlap_ids
    per_system_accs: List[List[float]] = [[] for _ in range(4)]
    per_article_rows: List[List] = []  # [id, acc1, acc2, acc3, acc4]

    for art_id in overlap_ids:
        row_vals = []
        ok = True
        for mi, m in enumerate(maps):
            val = load_accuracy_percent(m[art_id], verbose=args.verbose)
            if val is None or math.isnan(val) or math.isinf(val):
                ok = False
                break
            row_vals.append(val)
        if ok:
            for i in range(4):
                per_system_accs[i].append(row_vals[i])
            per_article_rows.append([art_id] + row_vals)

    # In theory all overlap_ids should parse, but guard anyway:
    n = len(per_article_rows)
    if n == 0:
        print("Overlapping IDs exist, but none had parsable accuracies.", file=sys.stderr)
        sys.exit(1)

    print(f"Usable overlapping pairs with valid accuracies: n = {n}")

    # Histograms (shared bins)
    edges = np.linspace(0, 100, args.bins + 1)
    counts_per_system = []
    for i in range(4):
        counts, _ = np.histogram(per_system_accs[i], bins=edges)
        counts_per_system.append(counts.tolist())

    # Report arrays for easy copy/paste
    print("\n=== Histogram edges (percent) ===")
    print([float(x) for x in edges.tolist()])

    for lbl, counts in zip(labels, counts_per_system):
        print(f"\n{lbl} counts (overlap-only):")
        print([int(c) for c in counts])

    # Summary stats
    print("\n=== Summary stats (overlap-only) ===")
    for lbl, accs in zip(labels, per_system_accs):
        mean, sd = mean_sd(accs)
        print(f"{lbl}: n={len(accs)}, mean={mean:.2f}%, sd={sd:.2f}%")

    # Optional CSV with per-article aligned rows
    if args.out_csv:
        try:
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id"] + labels)
                for row in per_article_rows:
                    w.writerow(row)
            print(f"\nPer-article overlapping accuracies written to: {args.out_csv}")
        except Exception as e:
            print(f"[warn] failed to write CSV '{args.out_csv}': {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
