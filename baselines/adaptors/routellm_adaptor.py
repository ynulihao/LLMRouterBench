"""RouteLLM data adaptor for LLMRouterBench.

This adaptor converts LLMRouterBench baseline records into the assets required by
the Matrix Factorization (MF) training pipeline in :mod:`baselines.RouteLLM`.

It performs two core tasks:
1. Builds pairwise comparison data between a specified strong/weak model pair.
2. Generates prompt embeddings aligned with the pairwise samples and saves them
   for MF training.

Outputs are saved under ``baselines/RouteLLM/data`` (configurable) and include:

``pairwise_train.json``
    Training comparisons with fields ``idx``, ``model_a``, ``model_b``, ``winner``...

``pairwise_test.json``
    Test comparisons in the same format (``winner`` may be ``model_a``, ``model_b``,
    or ``tie`` when scores match).

``prompt_index.json``
    Mapping from ``idx`` back to dataset/prompt metadata for reproducibility.

``prompt_embeddings.npy``
    Float32 NumPy array (``num_prompts x dim``) of embeddings for each ``idx``.

``metadata.json``
    Summary of the conversion (model pair, counts, embedding dim, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from generators.factory import create_generator

# Re-use validation helper to ensure sensible train/test ratios
from .common import validate_train_ratio


class RouteLLMAdaptor:
    """Adaptor that prepares RouteLLM MF training assets from baseline data."""

    def __init__(
        self,
        config_path: str,
        strong_model: str,
        weak_model: str,
        random_seed: int = 42,
        train_ratio: float = 0.8,
        embedding_config_path: str = "config/embedding_config.yaml",
        ood_datasets: Optional[List[str]] = None,
    ) -> None:
        validate_train_ratio(train_ratio)

        self.config_path = config_path
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.embedding_config_path = embedding_config_path
        self.ood_datasets = ood_datasets or []

        self.loader = BaselineDataLoader(config_path=config_path)
        self._validate_model_filters()

        # These are populated during conversion
        self.prompt_info: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.prompt_keys: List[Tuple[str, str]] = []
        self.prompt_to_idx: Dict[Tuple[str, str], int] = {}

        self.embedding_generator = None
        self._initialize_embedding_generator()

        logger.info(
            "Initialized RouteLLM adaptor with seed={}, ratio={:.2f}, pair={} vs {}",
            random_seed,
            train_ratio,
            strong_model,
            weak_model,
        )
        if self.ood_datasets:
            logger.info("OOD datasets specified: {}", self.ood_datasets)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def convert(self, output_dir: str = "baselines/RouteLLM/data") -> Dict[str, str]:
        """Execute the conversion pipeline.

        Args:
            output_dir: Base directory where artefacts will be written.

        Returns:
            Mapping of artefact name to the file path on disk.
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Load and inspect baseline records
        # ------------------------------------------------------------------
        logger.info("Loading baseline records...")
        all_records = self.loader.load_all_records()
        logger.info("Loaded {} records", len(all_records))

        if not all_records:
            raise ValueError("No baseline records found. Check baseline configuration.")

        self._build_prompt_catalog(all_records)
        if not self.prompt_keys:
            raise ValueError(
                "No prompts include both the strong and weak model with distinct scores."
            )

        logger.info(
            "Identified {} prompts suitable for {} vs {} comparisons",
            len(self.prompt_keys),
            self.strong_model,
            self.weak_model,
        )

        # ------------------------------------------------------------------
        # Train/test split: re-use loader's prompt-wise split logic
        # ------------------------------------------------------------------
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

        initial_train_pairs = self._build_pairwise_samples(train_records, include_ties=False)
        initial_test_pairs = self._build_pairwise_samples(test_records, include_ties=True)

        logger.info("Generated {} train pairwise samples", len(initial_train_pairs))
        logger.info("Generated {} test pairwise samples", len(initial_test_pairs))

        if not initial_train_pairs and not initial_test_pairs:
            raise ValueError(
                "No non-tied comparisons were generated. Ensure the selected model"
                " pair has differing scores on at least one prompt."
            )

        # Only keep prompts that yielded a comparison in either split
        used_prompt_indices = {
            sample["idx"] for sample in (*initial_train_pairs, *initial_test_pairs)
        }
        if not used_prompt_indices:
            raise ValueError("No usable prompts after removing ties. Aborting conversion.")

        self._prune_prompt_catalog(used_prompt_indices)
        logger.info("After pruning, {} prompts remain", len(self.prompt_keys))

        # Rebuild pairwise samples with the updated prompt indices
        train_pairs = self._build_pairwise_samples(train_records, include_ties=False)
        test_pairs = self._build_pairwise_samples(test_records, include_ties=True)

        self._assert_test_coverage(test_records, test_pairs)

        logger.info("Final train pairwise samples: {}", len(train_pairs))
        logger.info("Final test pairwise samples: {}", len(test_pairs))

        if not train_pairs and not test_pairs:
            raise ValueError("No samples remain after pruning unused prompts.")

        # ------------------------------------------------------------------
        # Prepare output directory (seed/split/model pair)
        # ------------------------------------------------------------------
        seed_str = f"seed{self.random_seed}"
        split_str = f"split{self.train_ratio:.1f}"
        pair_str = f"{self._sanitize_name(self.strong_model)}__vs__{self._sanitize_name(self.weak_model)}"
        output_path = output_path / f"{seed_str}_{split_str}_{pair_str}"
        output_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Write artefacts
        # ------------------------------------------------------------------
        train_file = output_path / "pairwise_train.json"
        test_file = output_path / "pairwise_test.json"
        prompt_index_file = output_path / "prompt_index.json"
        embeddings_file = output_path / "prompt_embeddings.npy"
        metadata_file = output_path / "metadata.json"

        self._write_json(train_file, train_pairs)
        self._write_json(test_file, test_pairs)
        self._write_json(prompt_index_file, self._build_prompt_index_payload())

        embeddings = self._generate_embeddings()
        np.save(embeddings_file, embeddings)
        logger.info(
            "Saved embeddings array with shape {} to {}", embeddings.shape, embeddings_file
        )

        metadata = {
            "strong_model": self.strong_model,
            "weak_model": self.weak_model,
            "train_samples": len(train_pairs),
            "test_samples": len(test_pairs),
            "num_prompts": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1] if embeddings.ndim == 2 else 0,
            "source_config": self.config_path,
            "embedding_config": self.embedding_config_path,
            "ood_datasets": self.ood_datasets,
        }
        self._write_json(metadata_file, metadata)

        logger.info("RouteLLM conversion complete!")

        return {
            "pairwise_train": str(train_file),
            "pairwise_test": str(test_file),
            "prompt_index": str(prompt_index_file),
            "embeddings": str(embeddings_file),
            "metadata": str(metadata_file),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_model_filters(self) -> None:
        """Ensure strong/weak models are permitted by the baseline config."""

        models_filter = self.loader.filters.get("models")
        if not models_filter:
            return  # No whitelist in effect

        missing = [
            model
            for model in {self.strong_model, self.weak_model}
            if model not in models_filter
        ]
        if missing:
            raise ValueError(
                "Baseline configuration disallows the configured strong/weak models. "
                f"Update {self.config_path} filters.models to include: {', '.join(sorted(missing))}"
            )

    def _initialize_embedding_generator(self) -> None:
        """Load embedding configuration and build an :class:`EmbeddingGenerator`."""
        try:
            with open(self.embedding_config_path, "r", encoding="utf-8") as fp:
                config = yaml.safe_load(fp)

            model_config = dict(config.get("embedding_model", {}))
            cache_config = config.get("cache")

            if not model_config:
                raise ValueError("embedding_config.yaml must define 'embedding_model'.")

            api_key = model_config.get("api_key", "")
            if api_key and api_key.isupper() and "_" in api_key:
                # Interpret uppercase values with underscore as environment variable names
                model_config["api_key"] = os.getenv(api_key, api_key)

            self.embedding_generator = create_generator(model_config, cache_config)
            logger.info(
                "Initialized EmbeddingGenerator with model {}",
                model_config.get("api_model_name", model_config.get("name")),
            )
        except Exception as exc:
            logger.error("Failed to initialise embedding generator: {}", exc)
            raise

    def _build_prompt_catalog(self, records: Iterable[BaselineRecord]) -> None:
        """Detect prompts containing the strong/weak models (ties retained for eval)."""

        prompt_groups: Dict[Tuple[str, str], List[BaselineRecord]] = defaultdict(list)
        for record in records:
            prompt_groups[(record.dataset_id, record.prompt)].append(record)

        eligible: Dict[Tuple[str, str], Dict[str, object]] = {}
        skipped_missing = 0
        tie_prompts = 0

        for key, prompt_records in prompt_groups.items():
            models = {r.model_name: r for r in prompt_records}
            strong = models.get(self.strong_model)
            weak = models.get(self.weak_model)

            if not strong or not weak:
                skipped_missing += 1
                continue

            is_tie = strong.score == weak.score
            if is_tie:
                tie_prompts += 1

            record_index = min(r.record_index for r in prompt_records)
            eligible[key] = {
                "dataset_id": key[0],
                "prompt": key[1],
                "record_index": record_index,
                "origin_query": strong.origin_query or weak.origin_query,
                "ground_truth": strong.ground_truth or weak.ground_truth,
                "is_tie": is_tie,
            }

        if skipped_missing:
            logger.info(
                "Skipped {} prompts because one of {}/{} was missing",
                skipped_missing,
                self.strong_model,
                self.weak_model,
            )
        if tie_prompts:
            logger.info(
                "Found {} prompts where {} and {} tie; kept for evaluation",
                tie_prompts,
                self.strong_model,
                self.weak_model,
            )

        # Sort prompts deterministically by dataset then record index
        sorted_items = sorted(
            eligible.items(), key=lambda item: (item[0][0], item[1]["record_index"])
        )

        self.prompt_keys = [key for key, _ in sorted_items]
        self.prompt_info = {key: info for key, info in sorted_items}
        self.prompt_to_idx = {key: idx for idx, key in enumerate(self.prompt_keys)}

    def _prune_prompt_catalog(self, used_indices: set[int]) -> None:
        """Drop prompts that never appear in train/test samples to keep indices dense.

        Caller is responsible for rebuilding pairwise samples after pruning so the
        new index mapping takes effect.
        """

        if len(used_indices) == len(self.prompt_keys):
            return  # Nothing to prune

        new_prompt_keys = []
        new_prompt_info: Dict[Tuple[str, str], Dict[str, object]] = {}
        new_prompt_to_idx: Dict[Tuple[str, str], int] = {}

        for old_idx, key in enumerate(self.prompt_keys):
            if old_idx not in used_indices:
                continue
            new_idx = len(new_prompt_keys)
            new_prompt_keys.append(key)
            new_prompt_info[key] = self.prompt_info[key]
            new_prompt_to_idx[key] = new_idx

        self.prompt_keys = new_prompt_keys
        self.prompt_info = new_prompt_info
        self.prompt_to_idx = new_prompt_to_idx

    def _build_pairwise_samples(
        self, records: Iterable[BaselineRecord], include_ties: bool
    ) -> List[Dict[str, object]]:
        """Create pairwise comparisons from a collection of records."""

        prompt_groups: Dict[Tuple[str, str], List[BaselineRecord]] = defaultdict(list)
        for record in records:
            key = (record.dataset_id, record.prompt)
            prompt_groups[key].append(record)

        samples: List[Dict[str, object]] = []
        missing = tie = 0

        for key, prompt_records in prompt_groups.items():
            if key not in self.prompt_to_idx:
                continue  # Prompt not eligible (missing model or tie discovered earlier)

            models = {r.model_name: r for r in prompt_records}
            strong = models.get(self.strong_model)
            weak = models.get(self.weak_model)

            if not strong or not weak:
                missing += 1
                continue

            is_tie = strong.score == weak.score
            if is_tie:
                tie += 1
                if not include_ties:
                    continue
                winner = "tie"
            else:
                winner = "model_a" if strong.score > weak.score else "model_b"

            idx = self.prompt_to_idx[key]
            info = self.prompt_info[key]

            samples.append(
                {
                    "idx": idx,
                    "dataset_id": info["dataset_id"],
                    "record_index": info["record_index"],
                    "prompt": info["prompt"],
                    "origin_query": info.get("origin_query"),
                    "ground_truth": info.get("ground_truth"),
                    "model_a": self.strong_model,
                    "model_b": self.weak_model,
                    "score_model_a": strong.score,
                    "score_model_b": weak.score,
                    "prediction_model_a": strong.prediction,
                    "prediction_model_b": weak.prediction,
                    "response_model_a": self._to_serializable(strong.raw_output),
                    "response_model_b": self._to_serializable(weak.raw_output),
                    "cost_model_a": strong.cost,
                    "cost_model_b": weak.cost,
                    "is_tie": is_tie,
                    "winner": winner,
                }
            )

        if missing:
            logger.debug(
                "Encountered {} prompts missing one of the models during split processing",
                missing,
            )
        if tie:
            logger.debug(
                "Encountered {} prompts with ties during split processing (include_ties=%s)",
                include_ties,
            )

        samples.sort(key=lambda item: item["idx"])
        return samples

    def _assert_test_coverage(
        self,
        test_records: Iterable[BaselineRecord],
        test_pairs: List[Dict[str, object]],
    ) -> None:
        """Ensure every prompt with both models present in the test split is emitted."""

        expected_keys = self._prompts_with_model_pair(test_records)
        produced_keys = {self.prompt_keys[sample["idx"]] for sample in test_pairs}

        missing = expected_keys - produced_keys
        if missing:
            formatted = ", ".join(f"{dataset}:{prompt}" for dataset, prompt in sorted(missing))
            raise AssertionError(
                "Test set is missing pairwise samples for prompts with both models present: "
                f"{formatted}"
            )

    def _prompts_with_model_pair(
        self, records: Iterable[BaselineRecord]
    ) -> Set[Tuple[str, str]]:
        """Return prompt keys where both strong/weak models appear."""

        prompt_groups: Dict[Tuple[str, str], List[BaselineRecord]] = defaultdict(list)
        for record in records:
            prompt_groups[(record.dataset_id, record.prompt)].append(record)

        expected: Set[Tuple[str, str]] = set()
        for key, prompt_records in prompt_groups.items():
            models = {r.model_name for r in prompt_records}
            if {self.strong_model, self.weak_model}.issubset(models):
                expected.add(key)
        return expected

    def _build_prompt_index_payload(self) -> List[Dict[str, object]]:
        """Create serialisable metadata for each prompt index."""

        payload = []
        for idx, key in enumerate(self.prompt_keys):
            info = self.prompt_info[key]
            payload.append(
                {
                    "idx": idx,
                    "dataset_id": info["dataset_id"],
                    "prompt": info["prompt"],
                    "origin_query": info.get("origin_query"),
                    "ground_truth": info.get("ground_truth"),
                    "record_index": info["record_index"],
                    "is_tie": info.get("is_tie", False),
                }
            )
        return payload

    def _generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for each prompt in ``prompt_keys`` order."""

        if self.embedding_generator is None:
            raise RuntimeError("Embedding generator is not initialised")

        raw_vectors: List[List[float]] = []
        embedding_dim: Optional[int] = None

        for key in tqdm(self.prompt_keys, desc="Generating embeddings"):
            prompt_text = self.prompt_info[key]["prompt"]
            try:
                result = self.embedding_generator.generate_embedding(prompt_text)
                vector = result.embeddings or []
            except Exception as exc:
                logger.error("Failed to generate embedding: {}", exc)
                vector = []

            if vector:
                if embedding_dim is None:
                    embedding_dim = len(vector)
                elif len(vector) != embedding_dim:
                    logger.warning(
                        "Embedding dimensionality changed from {} to {}; truncating",
                        embedding_dim,
                        len(vector),
                    )
                    vector = vector[:embedding_dim]

            raw_vectors.append(vector)

        if embedding_dim is None:
            raise RuntimeError(
                "Failed to obtain any embeddings. Check embedding configuration and API."
            )

        zero_vector = [0.0] * embedding_dim
        vectors = []
        for vector in raw_vectors:
            if not vector:
                vectors.append(zero_vector)
            elif len(vector) != embedding_dim:
                vectors.append(vector[:embedding_dim])
            else:
                vectors.append(vector)

        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        logger.info("Wrote {}", path)

    @staticmethod
    def _to_serializable(value: object) -> object:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return json.loads(value) if isinstance(value, str) else str(value)
        except Exception:
            return str(value)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("/", "_").replace(" ", "_")


def main() -> None:
    """Command-line entry point for the RouteLLM adaptor."""

    parser = argparse.ArgumentParser(
        description="Convert LLMRouterBench baseline data to RouteLLM MF training format"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline_config.yaml",
        help="Path to baseline configuration file",
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        required=True,
        help="Name of the strong model (must exist in baseline records)",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        required=True,
        help="Name of the weak model (must exist in baseline records)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt-level train/test split",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Proportion of prompts assigned to the training set (0-1)",
    )
    parser.add_argument(
        "--embedding-config",
        type=str,
        default="config/embedding_config.yaml",
        help="Path to embedding model configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/RouteLLM/data",
        help="Directory where RouteLLM artefacts will be saved",
    )
    parser.add_argument(
        "--ood-datasets",
        type=str,
        default="",
        help="Comma-separated dataset IDs to treat as out-of-domain (all go to test)",
    )

    args = parser.parse_args()

    ood_datasets = [d.strip() for d in args.ood_datasets.split(",") if d.strip()]

    adaptor = RouteLLMAdaptor(
        config_path=args.config,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        random_seed=args.seed,
        train_ratio=args.split_ratio,
        embedding_config_path=args.embedding_config,
        ood_datasets=ood_datasets,
    )

    outputs = adaptor.convert(output_dir=args.output_dir)

    logger.info("Generated artefacts:")
    for key, path in outputs.items():
        logger.info("  {}: {}", key, path)


if __name__ == "__main__":
    main()
