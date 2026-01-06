"""
Model-SAT data adaptor for LLMRouterBench.

Converts baseline benchmark results to the JSON format expected by the
Model-SAT training pipeline.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from .common import fill_missing_models_scores, get_unique_models, log_filled_statistics


class ModelSATAdaptor:
    """
    Adaptor for converting baseline data to Model-SAT format.

    Model-SAT expects per-dataset JSON files for each split:
        [
            {
                "index": int,
                "question": str,
                "scores": {model_name: score, ...}
            },
            ...
        ]

    Train and test files are generated separately for every dataset that appears
    in the baseline cache.
    """

    def __init__(
        self,
        config_path: str,
        random_seed: int = 42,
        train_ratio: float = 0.8,
        ood_datasets: Optional[List[str]] = None,
    ):
        """
        Initialize Model-SAT adaptor.

        Args:
            config_path: Path to baseline configuration YAML file.
            random_seed: Random seed for train/test splitting.
            train_ratio: Proportion of prompts assigned to the train split.
            ood_datasets: Optional list of dataset IDs treated as OOD
                          (entirely routed to the test split).
        """
        self.config_path = config_path
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.ood_datasets = ood_datasets or []
        self.loader = BaselineDataLoader(config_path=config_path)

        logger.info(
            "Initialized Model-SAT adaptor with seed={}, train_ratio={:.2f}",
            random_seed,
            train_ratio,
        )
        if self.ood_datasets:
            logger.info("Treating OOD datasets as test-only: {}", self.ood_datasets)

    def convert(
        self,
        output_dir: str = "baselines/MODEL-SAT/original_data",
        subset_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Convert baseline data to Model-SAT JSON files.

        Args:
            output_dir: Base directory for generated files.
            subset_name: Optional name for the subset directory. Defaults to
                         ``f\"subset_logs_{random_seed}\"`` when omitted.

        Returns:
            Mapping ``{"train": {dataset: path}, "test": {dataset: path}}``.
        """
        output_path = Path(output_dir)
        subset_dir_name = subset_name or f"subset_logs_{self.random_seed}"
        subset_path = output_path / subset_dir_name
        subset_path.mkdir(parents=True, exist_ok=True)

        logger.info("Writing Model-SAT data to {}", subset_path)

        # Load all baseline records
        all_records = self.loader.load_all_records()
        logger.info("Loaded {} baseline records", len(all_records))

        if not all_records:
            logger.warning("No records found – nothing to convert.")
            return {"train": {}, "test": {}}

        # Gather the universe of models to keep score dictionaries aligned
        all_models = get_unique_models(all_records)
        logger.info("Detected {} unique models", len(all_models))

        # Split into train/test without leaking prompts between splits
        train_records, test_records = self.loader.split_by_dataset_then_prompt(
            all_records,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            ood_datasets=self.ood_datasets,
        )
        logger.info(
            "Split baseline cache into {} train and {} test records",
            len(train_records),
            len(test_records),
        )

        # Convert each split
        train_datasets, train_counter = self._convert_split(train_records, all_models)
        test_datasets, test_counter = self._convert_split(test_records, all_models)

        # Log statistics about missing-score filling
        log_filled_statistics(train_counter, prefix="[TRAIN] ")
        log_filled_statistics(test_counter, prefix="[TEST] ")

        # Write JSON files
        written_files: Dict[str, Dict[str, str]] = {"train": {}, "test": {}}
        all_dataset_ids = sorted(set(train_datasets.keys()) | set(test_datasets.keys()))
        for dataset_id in all_dataset_ids:
            train_items = train_datasets.get(dataset_id, [])
            test_items = test_datasets.get(dataset_id, [])

            if train_items:
                train_file = subset_path / f"{dataset_id}_train.json"
                self._write_json(train_file, train_items)
                written_files["train"][dataset_id] = str(train_file)
                logger.info(
                    "Wrote {} train prompts for dataset {} to {}",
                    len(train_items),
                    dataset_id,
                    train_file,
                )

            if test_items:
                test_file = subset_path / f"{dataset_id}_test.json"
                self._write_json(test_file, test_items)
                written_files["test"][dataset_id] = str(test_file)
                logger.info(
                    "Wrote {} test prompts for dataset {} to {}",
                    len(test_items),
                    dataset_id,
                    test_file,
                )

            if not train_items and not test_items:
                logger.warning(
                    "Dataset {} produced no records in either split; skipping file generation.",
                    dataset_id,
                )

        logger.info("Model-SAT conversion complete!")
        return written_files

    def _convert_split(
        self,
        records: List[BaselineRecord],
        all_models: List[str],
    ) -> Tuple[Dict[str, List[Dict]], Counter]:
        """
        Convert records belonging to a single split into Model-SAT entries.

        Returns:
            Tuple of (per-dataset entries, filled-model counter).
        """
        per_dataset: Dict[str, List[Dict]] = {}
        filled_counter: Counter = Counter()

        dataset_to_records: Dict[str, List[BaselineRecord]] = defaultdict(list)
        for record in records:
            dataset_to_records[record.dataset_id].append(record)

        for dataset_id, dataset_records in dataset_to_records.items():
            entries, dataset_filled = self._convert_dataset_records(
                dataset_records,
                all_models,
            )
            per_dataset[dataset_id] = entries
            filled_counter.update(dataset_filled)

        return per_dataset, filled_counter

    def _convert_dataset_records(
        self,
        records: List[BaselineRecord],
        all_models: List[str],
    ) -> Tuple[List[Dict], Counter]:
        """
        Convert all records for a dataset into Model-SAT prompts.

        Returns:
            Tuple of (prompt entries, filled-model counter).
        """
        prompt_groups: Dict[str, List[BaselineRecord]] = defaultdict(list)
        for record in records:
            prompt_groups[record.prompt].append(record)

        entries: List[Dict] = []
        filled_counter: Counter = Counter()

        for prompt, prompt_records in prompt_groups.items():
            # Use the minimum record index as the canonical index for the prompt.
            index = min(r.record_index for r in prompt_records)
            scores = {r.model_name: r.score for r in prompt_records}

            # Ensure every known model has a score (fill missing ones with 0.0).
            filled = fill_missing_models_scores(scores, all_models, fill_value=0.0)
            filled_counter.update(filled)

            entries.append(
                {
                    "index": index,
                    "question": prompt,
                    "scores": scores,
                }
            )

        entries.sort(key=lambda item: item["index"])
        return entries, filled_counter

    @staticmethod
    def _write_json(file_path: Path, data: List[Dict]):
        """Write a JSON list with UTF-8 encoding and pretty formatting."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the adaptor."""
    parser = argparse.ArgumentParser(
        description="Convert LLMRouterBench baseline cache to Model-SAT JSON files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline_config.yaml",
        help="Path to the baseline loader configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt splitting (used in subset directory name).",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Proportion of prompts assigned to the train split (0-1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/MODEL-SAT/original_data",
        help="Base directory where the subset folder is created.",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Optional name for the subset directory inside output-dir.",
    )
    parser.add_argument(
        "--ood-datasets",
        type=str,
        default="",
        help="Comma-separated dataset IDs that should be treated as OOD (test-only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ood_datasets = []
    if args.ood_datasets:
        ood_datasets = [dataset.strip() for dataset in args.ood_datasets.split(",") if dataset.strip()]

    adaptor = ModelSATAdaptor(
        config_path=args.config,
        random_seed=args.seed,
        train_ratio=args.split_ratio,
        ood_datasets=ood_datasets,
    )

    written_files = adaptor.convert(output_dir=args.output_dir, subset_name=args.subset_name)

    logger.info("Generated Model-SAT files:")
    for split, mapping in written_files.items():
        if not mapping:
            logger.info("  {}: (no files generated)", split)
            continue
        for dataset_id, path in sorted(mapping.items()):
            logger.info("  {} | {} -> {}", split.upper(), dataset_id, path)


if __name__ == "__main__":
    main()
