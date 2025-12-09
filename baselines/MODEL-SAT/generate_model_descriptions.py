#!/usr/bin/env python3
"""
Compute per-category accuracies from an evaluation log and write a
natural-language description JSON.

Example:
    python genetate_model_descriptions.py \
        --input  mmlupro_train.json \
        --output model_description.json
"""
from __future__ import annotations

import argparse
import json
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Regex used to extract the category name from each question
CATEGORY_RE = re.compile(r"Answer the following (.+?) question")


# ────────────────────────── core logic ──────────────────────────
def extract_category(question: str) -> str:
    """Return the category name embedded in the question string."""
    match = CATEGORY_RE.search(question)
    if not match:
        raise ValueError(f"Category name not found in: {question[:80]}…")
    return match.group(1)


def compute_accuracy(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy for every (category, model) pair.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: {category: {model: accuracy}}
    """
    # counts[category][model] = [correct, total]
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for item in records:
        category = extract_category(item["question"])
        for model_name, score in item["scores"].items():
            correct, total = counts[category][model_name]
            counts[category][model_name] = [correct + (score == 1.0), total + 1]

    return {
        cat: {m: c / t for m, (c, t) in models.items()}
        for cat, models in counts.items()
    }


def build_descriptions(acc: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Convert accuracy numbers into plain-English sentences.

    Returns
    -------
    Dict[str, str]
        {model: description sentence}
    """
    all_models = sorted({m for v in acc.values() for m in v})
    categories = sorted(acc)

    return {
        model: "The model achieves "
        + ", ".join(
            f"accuracy {acc[cat].get(model, 0) * 100:.1f}% on the task of {cat}"
            for cat in categories
        )
        + "."
        for model in all_models
    }


# ────────────────────────── CLI handling ──────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracies and write summary JSON."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Evaluation JSON file."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Destination for the model description JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Read the evaluation log
    with args.input.open() as f:
        records = json.load(f)

    # Build descriptions
    descriptions = build_descriptions(compute_accuracy(records))

    # Write JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(descriptions, f, indent=4)


if __name__ == "__main__":
    main()