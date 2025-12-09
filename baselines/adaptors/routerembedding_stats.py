"""
Utility script to inspect RouterEmbedding training data scale.

Given a ``train_prompts.jsonl`` file (output of RouterEmbeddingAdaptor),
the script reports how many anchor-positive pairs would be produced per
dataset under the current pairing logic, along with the potential number
of negative attachments.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate RouterEmbedding pair counts from train_prompts.jsonl"
    )
    parser.add_argument(
        "--train-prompts",
        type=Path,
        required=True,
        help="Path to RouterEmbeddingAdaptor's train_prompts.jsonl file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the statistics as JSON.",
    )
    parser.add_argument(
        "--top-k-negatives",
        type=int,
        default=32,
        help="Cap on negatives per sample when estimating dataset size.",
    )
    parser.add_argument(
        "--max-pos-per-anchor",
        type=int,
        default=1,
        help="Maximum number of positives per anchor to mirror adaptor behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_prompts.exists():
        raise FileNotFoundError(f"{args.train_prompts} not found.")
    if args.top_k_negatives < 0:
        raise ValueError("top_k_negatives must be non-negative")
    if args.max_pos_per_anchor < 0:
        raise ValueError("max_pos_per_anchor must be non-negative")

    stats = compute_pair_statistics(
        args.train_prompts,
        top_k=args.top_k_negatives,
        max_pos=args.max_pos_per_anchor,
    )
    total_pairs = sum(info["pair_count"] for info in stats.values())
    total_negative_mentions = sum(info["estimated_negative_assignments"] for info in stats.values())
    total_potential_negatives = sum(info["potential_negative_assignments"] for info in stats.values())

    logger.info("===== RouterEmbedding Pair Estimate =====")
    logger.info("Total datasets: {}", len(stats))
    logger.info("Total prompts: {}", sum(info["total_prompts"] for info in stats.values()))
    logger.info("Total potential pairs: {}", total_pairs)
    logger.info(
        "Estimated negative assignments (top_k={}, max_pos={}): {}",
        args.top_k_negatives,
        args.max_pos_per_anchor,
        total_negative_mentions,
    )
    logger.info("Upper-bound negative assignments (no cap): {}", total_potential_negatives)

    for dataset_id, info in sorted(stats.items(), key=lambda item: item[0]):
        logger.info(
            "[{}] prompts={}, pairs={}, est_negatives={}, potential_negatives={}",
            dataset_id,
            info["total_prompts"],
            info["pair_count"],
            info["estimated_negative_assignments"],
            info["potential_negative_assignments"],
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": {
                        "datasets": len(stats),
                        "total_prompts": sum(info["total_prompts"] for info in stats.values()),
                        "total_pairs": total_pairs,
                        "estimated_negative_assignments": total_negative_mentions,
                        "potential_negative_assignments": total_potential_negatives,
                        "top_k_negatives": args.top_k_negatives,
                        "max_pos_per_anchor": args.max_pos_per_anchor,
                    },
                    "per_dataset": stats,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Wrote JSON summary to {}", args.output_json)


def compute_pair_statistics(train_prompts_path: Path, top_k: int, max_pos: int) -> Dict[str, Dict[str, int]]:
    """
    Compute per-dataset pair counts following the adaptor's logic.
    """
    dataset_groups: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    dataset_totals: Dict[str, int] = defaultdict(int)

    with train_prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            dataset_id = record.get("dataset_id")
            models = record.get("models", [])
            correct = sum(1 for entry in models if entry.get("correct"))
            total = len(models)
            accuracy_key = (correct, total)

            dataset_totals[dataset_id] += 1
            dataset_groups[dataset_id][accuracy_key] += 1

    stats: Dict[str, Dict[str, int]] = {}
    for dataset_id, acc_counts in dataset_groups.items():
        total_prompts = dataset_totals[dataset_id]
        pair_count = 0
        potential_negative_assignments = 0
        estimated_negative_assignments = 0

        for (correct, total_models), count in acc_counts.items():
            if count < 2 or max_pos == 0:
                continue
            group_pairs = _estimate_pairs_for_group(count, max_pos)
            pair_count += group_pairs

            negative_options = max(total_prompts - count, 0)
            potential_negative_assignments += group_pairs * negative_options
            capped_negatives = group_pairs * min(negative_options, top_k)
            estimated_negative_assignments += capped_negatives

        stats[dataset_id] = {
            "total_prompts": total_prompts,
            "pair_count": pair_count,
            "estimated_negative_assignments": estimated_negative_assignments,
            "potential_negative_assignments": potential_negative_assignments,
        }

    return stats


def _estimate_pairs_for_group(group_size: int, max_pos: int) -> int:
    """
    Mirror adaptor's per-group pairing strategy with cap per anchor.
    """
    if max_pos <= 0 or group_size < 2:
        return 0

    total_pairs = 0
    for i in range(group_size - 1):
        remaining = group_size - i - 1
        total_pairs += min(max_pos, remaining)

    return total_pairs


if __name__ == "__main__":
    main()
