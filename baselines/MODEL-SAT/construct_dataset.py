#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge *_{train,test}.json files inside a directory and emit one file per
split.  Splits that are absent in the source directory are silently ignored,
and you only need to supply output paths for the splits you really want.

Both ordinary JSON lists **and** newline-delimited JSON (NDJSON) inputs are
supported.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Mapping, MutableMapping, Set

# ──────────────────────────────────────────────────────────
SPLIT_SUFFIXES = {
    "_train.json": "train",
    "_test.json":  "test",
}


# ──────────────────────────────────────────────────────────
def load_json(path: str) -> List[Dict]:
    """Load either a JSON list or NDJSON file."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def build_unified_json(
    data_dir: str,
    out_paths: Mapping[str, str],
    threshold: float = 0.0,
    ood_out: str | None = None,
) -> None:
    """
    Merge *_{train,test}.json files into unified per-split files.

    Parameters
    ----------
    data_dir : str
        Directory containing the source files.
    out_paths : Mapping[str, str]
        Mapping ``{split: destination_path}``.  Supply only the splits you want
        written.
    threshold : float, optional
        Score threshold above which ``is_correct_sc`` is ``True``.  Default 0.0.
    """
    buffers: MutableMapping[str, List[Dict]] = defaultdict(list)
    counters:  MutableMapping[str, int]       = defaultdict(int)
    ood_buffer: List[Dict] = []

    pattern = os.path.join(data_dir, "*.json")

    # Pre-scan file names to detect available tasks and splits
    all_paths = sorted(glob.glob(pattern))
    basenames = [os.path.basename(p) for p in all_paths]

    task_splits: MutableMapping[str, Set[str]] = defaultdict(set)
    for base in basenames:
        for suff, s in SPLIT_SUFFIXES.items():
            if base.endswith(suff):
                task = re.sub(rf"_{s}\.json$", "", base)
                task_splits[task].add(s)

    ood_tasks = {t for t, splits in task_splits.items() if "test" in splits and "train" not in splits}

    for path in all_paths:
        base = os.path.basename(path)

        # Identify the split (train / test) by suffix
        split = next(
            (s for suff, s in SPLIT_SUFFIXES.items() if base.endswith(suff)),
            None,
        )
        if split is None:
            continue  # unrelated file

        task = re.sub(rf"_{split}\.json$", "", base)
        for sample in load_json(path):
            idx = counters[split]  # 0-based, continuous within each split

            for model, score in sample["scores"].items():
                record = {
                    "query":         sample["question"],
                    "model":         model,
                    "is_correct_sc": bool(score > threshold),
                    "task":          task,
                    "index":         idx,
                }
                if split == "test" and task in ood_tasks:
                    ood_buffer.append(record)
                else:
                    buffers[split].append(record)
            counters[split] += 1

    # ─── Write outputs ────────────────────────────────────
    for split, records in buffers.items():
        dst = out_paths.get(split)
        if not dst:  # user did not request this split
            print(f"∙ [skip] {split}: {len(records):,} records (no --{split}-out)")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(
            f"✔ Wrote {dst}  "
            f"({len(records):,} records, {counters[split]} questions)"
        )

    # Write OOD if any test-only tasks exist
    if ood_buffer:
        if not ood_out:
            # default location relative to this script
            ood_dir = os.path.join(os.path.dirname(__file__), "data", "ood")
            ood_out = os.path.join(ood_dir, "ood.json")
        else:
            ood_dir = os.path.dirname(ood_out)
        if ood_dir:
            os.makedirs(ood_dir, exist_ok=True)
        with open(ood_out, "w", encoding="utf-8") as f:
            json.dump(ood_buffer, f, ensure_ascii=False, indent=2)
        print(f"✔ Wrote OOD {ood_out} ({len(ood_buffer):,} records)")

    missing = [s for s in out_paths if s not in buffers]
    if missing:
        print(
            "\nWarning: requested split(s) with no matching source files → "
            + ", ".join(missing)
        )


# ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(
        description="Merge *_{train,test}.json files into unified files."
    )
    p.add_argument("-d", "--data-dir", required=True, help="Source directory.")
    p.add_argument("--train-out", help="Output path for the merged train split.")
    p.add_argument("--test-out",  help="Output path for the merged test split.")
    p.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Score threshold for correctness (default: 0.0).",
    )
    p.add_argument(
        "--ood-out",
        help="Output path for OOD records (test-only tasks). If omitted, defaults to baselines/MODEL-SAT/data/ood/ood.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = {
        split: path
        for split, path in (
            ("train", args.train_out),
            ("test",  args.test_out),
        )
        if path  # keep only non-None
    }

    # Keep behavior: require at least one output unless OOD-only desired.
    # Since we now emit OOD per task automatically, it's reasonable to allow
    # the user to run with only --ood-out as well.
    if not out_paths and not args.ood_out:
        raise SystemExit(
            "ERROR: No output paths supplied. Use --train-out/--test-out and/or --ood-out."
        )

    build_unified_json(
        data_dir=args.data_dir,
        out_paths=out_paths,
        threshold=args.threshold,
        ood_out=args.ood_out,
    )


if __name__ == "__main__":
    main()
