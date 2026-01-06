"""
RouterEmbedding adaptor for LLMRouterBench.

This adaptor prepares prompt-level JSONL datasets tailored for downstream
embedding-model training. It reuses the shared baseline loader to split
records by dataset and prompt, ensuring that all model answers to the same
question stay within one split and preventing leakage between train/test.

Usage:
    python -m baselines.adaptors.routerembedding_adaptor \
        --config config/baseline_config.yaml \
        --output-dir baselines/RouterEmbedding/data \
        --train-ratio 0.8 \
        --random-seed 42 \
        --ood-datasets mmlu,bbh
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord


class RouterEmbeddingAdaptor:
    """
    Convert unified baseline records into RouterEmbedding-ready datasets.

    The adaptor focuses on producing two JSONL files (train/test). Each line
    groups all model evaluations for a single prompt along with correctness
    metadata so that embedding pipelines can easily learn from per-question
    outcomes.
    """

    def __init__(
        self,
        config_path: str,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        ood_datasets: Optional[List[str]] = None,
        force_overwrite: bool = False,
        top_k_negatives: int = 32,
        max_pos_per_anchor: int = 1,
    ) -> None:
        self.config_path = config_path
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.ood_datasets = ood_datasets or []
        self.force_overwrite = force_overwrite
        self.top_k_negatives = top_k_negatives
        self.max_pos_per_anchor = max_pos_per_anchor

        self.loader = BaselineDataLoader(config_path=config_path)
        self._rng = random.Random(random_seed)

        if self.top_k_negatives < 0:
            raise ValueError("top_k_negatives must be non-negative")
        if self.max_pos_per_anchor < 0:
            raise ValueError("max_pos_per_anchor must be non-negative")

        if self.ood_datasets:
            logger.info(
                "RouterEmbedding adaptor will treat OOD datasets as test-only: {}", self.ood_datasets
            )

    def convert(self, output_dir: str = "baselines/RouterEmbedding/data") -> Dict[str, str]:
        """
        Build RouterEmbedding datasets and return generated file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Loading baseline records from {}", self.config_path)
        all_records = self.loader.load_all_records()
        logger.info("Loaded {} baseline records", len(all_records))

        if not all_records:
            raise ValueError("No baseline records found; cannot create RouterEmbedding dataset.")

        train_records, test_records = self.loader.split_by_dataset_then_prompt(
            all_records,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            ood_datasets=self.ood_datasets,
        )

        logger.info(
            "Split complete: {} train records, {} test records",
            len(train_records),
            len(test_records),
        )

        seed_str = f"seed{self.random_seed}"
        split_str = f"split{self.train_ratio:.1f}"
        subset_dir = output_path / f"{seed_str}_{split_str}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        train_prompts = self._aggregate_prompts(train_records)
        test_prompts = self._aggregate_prompts(test_records)

        train_file = subset_dir / "train_prompts.jsonl"
        test_file = subset_dir / "test_prompts.jsonl"

        self._maybe_write_jsonl(train_file, train_prompts, label="train split")
        self._maybe_write_jsonl(test_file, test_prompts, label="test split")

        # Further split training prompts into correctness-based subsets
        subsets = self._split_train_prompts(train_prompts)
        train_subsets_dir = subset_dir / "train_subsets"
        train_subsets_dir.mkdir(parents=True, exist_ok=True)

        subset_files = {}
        subset_counts = {}
        for subset_name, subset_prompts in subsets.items():
            subset_path = train_subsets_dir / f"{subset_name}.jsonl"
            self._maybe_write_jsonl(subset_path, subset_prompts, label=f"train subset '{subset_name}'")
            subset_files[subset_name] = str(subset_path)
            subset_counts[subset_name] = len(subset_prompts)

        logger.info("Train subset stats: {}", subset_counts)

        # Build accuracy-aware pairs for embedding training
        pair_dir = subset_dir / "train_pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_file = pair_dir / "accuracy_triplets.jsonl"

        # TODO: Consider generating triplets from specific subsets (e.g., all_correct)
        # instead of the full train set if future experiments require that granularity.
        self._build_accuracy_pairs(train_prompts, pair_file)

        logger.info(
            "RouterEmbedding dataset prepared with {} train prompts and {} test prompts",
            len(train_prompts),
            len(test_prompts),
        )

        return {
            "train": str(train_file),
            "test": str(test_file),
            "train_subsets": subset_files,
            "train_pairs": str(pair_file),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _aggregate_prompts(self, records: List[BaselineRecord]) -> List[Dict[str, object]]:
        """
        Aggregate model answers by prompt so downstream embedding training can
        see every model's performance for the same question.
        """
        from collections import defaultdict

        grouped: Dict[Tuple[str, str], List[BaselineRecord]] = defaultdict(list)
        for record in records:
            key = (record.dataset_id, record.prompt)
            grouped[key].append(record)

        aggregated: List[Dict[str, object]] = []
        for (dataset_id, prompt), prompt_records in grouped.items():
            prompt_records.sort(key=lambda r: r.model_name)
            first_record = prompt_records[0]

            aggregated.append(
                {
                    "dataset_id": dataset_id,
                    "prompt": prompt,
                    "origin_query": first_record.origin_query,
                    "record_index": min(r.record_index for r in prompt_records),
                    "models": [
                        {
                            "model_name": r.model_name,
                            "score": r.score,
                            "correct": bool(r.score > 0),
                            "prediction": r.prediction,
                            "ground_truth": r.ground_truth,
                            "prompt_tokens": r.prompt_tokens,
                            "completion_tokens": r.completion_tokens,
                            "cost": r.cost,
                        }
                        for r in prompt_records
                    ],
                }
            )

        return aggregated

    def _write_jsonl(self, file_path: Path, payload: List[Dict[str, object]]) -> None:
        """Write aggregated prompts to a JSONL file."""
        count = 0
        with open(file_path, "w", encoding="utf-8") as handle:
            for item in payload:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

        logger.info("Wrote {} items to {}", count, file_path)

    def _maybe_write_jsonl(self, file_path: Path, payload: List[Dict[str, object]], label: str) -> None:
        """
        Optionally write a JSONL file depending on force_overwrite / cache state.
        """
        if file_path.exists() and not self.force_overwrite:
            logger.info(
                "{} already exists at {}; skipping write ({} items) because force_overwrite=False",
                label,
                file_path,
                len(payload),
            )
            return

        self._write_jsonl(file_path, payload)

    def _split_train_prompts(self, prompts: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        """
        Partition training prompts into all_correct, all_error, and normal subsets.
        """
        all_correct: List[Dict[str, object]] = []
        all_error: List[Dict[str, object]] = []
        normal: List[Dict[str, object]] = []

        for item in prompts:
            model_entries = item.get("models", [])
            if not model_entries:
                normal.append(item)
                continue

            correct_flags = [bool(entry.get("correct")) for entry in model_entries]

            if all(correct_flags):
                all_correct.append(item)
            elif not any(correct_flags):
                all_error.append(item)
            else:
                normal.append(item)

        return {
            "all_correct": all_correct,
            "all_error": all_error,
            "normal": normal,
        }

    def _build_accuracy_pairs(self, prompts: List[Dict[str, object]], output_path: Path) -> None:
        """
        Construct anchor-positive-negative samples grouped by dataset and accuracy.
        """
        if output_path.exists() and not self.force_overwrite:
            logger.info("Pair file {} already exists; skipping rebuild.", output_path)
            return

        grouped_by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for item in prompts:
            grouped_by_dataset[item["dataset_id"]].append(item)

        samples: List[Dict[str, object]] = []

        for dataset_id, dataset_prompts in tqdm(
            grouped_by_dataset.items(), desc="Building accuracy triplets", unit="dataset"
        ):
            accuracy_groups: Dict[float, List[Dict[str, object]]] = defaultdict(list)
            for prompt_item in dataset_prompts:
                accuracy = self._calculate_accuracy(prompt_item)
                prompt_item["_accuracy"] = accuracy
                accuracy_groups[accuracy].append(prompt_item)

            for accuracy_value, prompt_list in accuracy_groups.items():
                if len(prompt_list) < 2 or self.max_pos_per_anchor == 0:
                    continue

                shuffled_prompts = prompt_list.copy()
                self._rng.shuffle(shuffled_prompts)
                neg_candidates = self._collect_negative_candidates(
                    accuracy_groups, exclude_accuracy=accuracy_value
                )

                n = len(shuffled_prompts)
                for anchor_idx in range(n - 1):
                    anchor = shuffled_prompts[anchor_idx]
                    remaining = n - anchor_idx - 1
                    take = min(self.max_pos_per_anchor, remaining)
                    if take <= 0:
                        continue

                    for offset in range(1, take + 1):
                        positive = shuffled_prompts[anchor_idx + offset]
                        sample = self._build_sample(dataset_id, anchor, positive, neg_candidates)
                        samples.append(sample)

        if not samples:
            logger.warning("No accuracy-based pairs were generated; check training data coverage.")

        self._write_jsonl(output_path, samples)

    def _collect_negative_candidates(
        self,
        accuracy_groups: Dict[float, List[Dict[str, object]]],
        exclude_accuracy: float,
    ) -> List[Dict[str, object]]:
        """
        Gather negative prompts (different accuracy) for a dataset.
        """
        negatives: List[Dict[str, object]] = []
        for acc, items in accuracy_groups.items():
            if acc == exclude_accuracy:
                continue
            negatives.extend(items)
        return negatives

    def _sample_negatives(
        self,
        anchor: Dict[str, object],
        negatives: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """
        Stratified sampling of negatives by accuracy gap with ratios 5:3:2.
        """
        if not negatives or self.top_k_negatives == 0:
            return []

        anchor_acc = anchor.get("_accuracy", 0.0)
        candidates = []
        for item in negatives:
            diff = abs(anchor_acc - item.get("_accuracy", 0.0))
            candidates.append((diff, item))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0])
        total = len(candidates)
        top_k = min(self.top_k_negatives, total)

        # Split candidate list into 5:3:2 strata based on distance.
        near_cut = math.ceil(total * 0.5)
        mid_cut = math.ceil(total * 0.8)

        near_pool = [item for _, item in candidates[:near_cut]]
        mid_pool = [item for _, item in candidates[near_cut:mid_cut]]
        far_pool = [item for _, item in candidates[mid_cut:]]

        near_target = math.floor(top_k * 0.5)
        mid_target = math.floor(top_k * 0.3)
        far_target = top_k - near_target - mid_target

        strata = [
            (near_pool, near_target),
            (mid_pool, mid_target),
            (far_pool, far_target),
        ]

        selected: List[Dict[str, object]] = []
        remaining = top_k
        leftovers: List[List[Dict[str, object]]] = []

        for pool, target in strata:
            if remaining <= 0:
                leftovers.append(pool)
                continue

            take = min(target, remaining, len(pool))
            shuffled_pool = pool.copy()
            self._rng.shuffle(shuffled_pool)
            chosen = shuffled_pool[:take]
            selected.extend(chosen)
            remaining -= len(chosen)
            leftovers.append(shuffled_pool[take:])

        if remaining > 0:
            spillover = [item for leftover in leftovers for item in leftover]
            if spillover:
                self._rng.shuffle(spillover)
                selected.extend(spillover[:remaining])

        return selected

    def _build_sample(
        self,
        dataset_id: str,
        anchor: Dict[str, object],
        positive: Dict[str, object],
        negatives: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """
        Build a single triplet-style sample.
        """
        anchor_message = self._format_message(anchor)
        positive_message = self._format_message(positive)

        sample: Dict[str, object] = {
            "channel": dataset_id,
            "messages": [anchor_message],
            "positive_messages": [[positive_message]],
        }

        sampled_negatives = self._sample_negatives(anchor, negatives)
        if sampled_negatives:
            sample["negative_messages"] = [[self._format_message(n)] for n in sampled_negatives]

        return sample

    @staticmethod
    def _calculate_accuracy(prompt_item: Dict[str, object]) -> float:
        models = prompt_item.get("models", [])
        if not models:
            return 0.0
        correct = sum(1 for entry in models if entry.get("correct"))
        return correct / len(models)

    @staticmethod
    def _format_message(prompt_item: Dict[str, object]) -> Dict[str, object]:
        return {
            "role": "user",
            "content": prompt_item.get("prompt", ""),
            "accuracy": prompt_item.get("_accuracy"),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RouterEmbedding datasets from the baseline cache.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline_config.yaml",
        help="Path to the baseline data loader configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/RouterEmbedding/data",
        help="Directory where the output JSONL files will be stored.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of prompts assigned to the training split.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used for the split.")
    parser.add_argument(
        "--ood-datasets",
        type=str,
        default="",
        help="Comma-separated list of dataset_ids that should be treated as OOD (all prompts go to test).",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Regenerate output files even if they already exist.",
    )
    parser.add_argument(
        "--top-k-negatives",
        type=int,
        default=32,
        help="Maximum number of negatives to attach per sample (stratified 5:3:2).",
    )
    parser.add_argument(
        "--max-pos-per-anchor",
        type=int,
        default=1,
        help="Maximum number of positive matches each anchor can contribute.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ood_list = [d.strip() for d in args.ood_datasets.split(",") if d.strip()]

    adaptor = RouterEmbeddingAdaptor(
        config_path=args.config,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        ood_datasets=ood_list,
        force_overwrite=args.force_overwrite,
        top_k_negatives=args.top_k_negatives,
        max_pos_per_anchor=args.max_pos_per_anchor,
    )

    adaptor.convert(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
