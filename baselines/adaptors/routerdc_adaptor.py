"""
RouterDC data adaptor for LLMRouterBench.

Converts baseline benchmark results to RouterDC format (JSON files with clustering).
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from .common import split_by_dataset, fill_missing_models_scores, log_filled_statistics
from generators.factory import create_generator
from generators.generator import EmbeddingGenerator


class RouterDCAdaptor:
    """
    Adaptor for converting baseline data to RouterDC format.

    RouterDC requires JSON files for each dataset with the following structure:
    [
        {
            "index": int,
            "question": str,
            "scores": {model_name: score, ...},
            "cluster_id": int
        },
        ...
    ]

    Train and test sets are generated separately per dataset.
    Cluster IDs are assigned using K-means clustering on prompt embeddings.
    """

    def __init__(
        self,
        config_path: str,
        random_seed: int = 42,
        train_ratio: float = 0.8,
        n_clusters: int = 50,
        embedding_config_path: str = "config/embedding_config.yaml",
        ood_datasets: Optional[List[str]] = None
    ):
        """
        Initialize RouterDC adaptor.

        Args:
            config_path: Path to baseline configuration YAML file
            random_seed: Random seed for train/test splitting and clustering
            train_ratio: Proportion of data for training (0.0-1.0)
            n_clusters: Number of clusters for K-means
            embedding_config_path: Path to embedding model configuration YAML file
            ood_datasets: Optional list of dataset IDs to treat as OOD (all go to test)
        """
        self.config_path = config_path
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.n_clusters = n_clusters
        self.embedding_config_path = embedding_config_path
        self.ood_datasets = ood_datasets or []
        self.loader = BaselineDataLoader(config_path=config_path)

        # Load embedding configuration and initialize generator
        self._initialize_embedding_generator()

        logger.info(
            f"Initialized RouterDC adaptor with seed={random_seed}, "
            f"ratio={train_ratio}, n_clusters={n_clusters}, "
            f"embedding_config={embedding_config_path}"
        )
        if self.ood_datasets:
            logger.info(f"OOD datasets: {self.ood_datasets}")

    def convert(self, output_dir: str = "baselines/RouterDC/data") -> Dict[str, List[str]]:
        """
        Convert baseline data to RouterDC format with global clustering.

        Args:
            output_dir: Directory to save output JSON files

        Returns:
            Dictionary mapping 'train' and 'test' to lists of file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all records
        logger.info("Loading baseline records...")
        all_records = self.loader.load_all_records()
        logger.info(f"Loaded {len(all_records)} records")

        # Get all unique models from entire baseline
        from .common import get_unique_models
        all_models = get_unique_models(all_records)
        logger.info(f"Found {len(all_models)} unique models across all datasets: {all_models}")

        # Step 1: Split by dataset then prompt (unified logic)
        logger.info("Splitting records by dataset then prompt...")
        train_records, test_records = self.loader.split_by_dataset_then_prompt(
            all_records,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            ood_datasets=self.ood_datasets
        )
        logger.info(f"Split complete: {len(train_records)} train, {len(test_records)} test records")

        # Create output directory with seed and split ratio
        seed_str = f"seed{self.random_seed}"
        split_str = f"split{self.train_ratio:.1f}"
        output_path = output_path / f"{seed_str}_{split_str}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 2: Group by dataset and convert to RouterDC format
        all_dataset_data = {}

        # Group train records by dataset
        train_by_dataset = defaultdict(list)
        for r in train_records:
            train_by_dataset[r.dataset_id].append(r)

        # Group test records by dataset
        test_by_dataset = defaultdict(list)
        for r in test_records:
            test_by_dataset[r.dataset_id].append(r)

        # Convert each dataset to RouterDC format
        all_datasets = set(train_by_dataset.keys()) | set(test_by_dataset.keys())
        for dataset_id in sorted(all_datasets):
            logger.info(f"Converting dataset: {dataset_id}")
            train_data = self._convert_records_to_format(
                train_by_dataset.get(dataset_id, []), all_models, dataset_id, "train"
            )
            test_data = self._convert_records_to_format(
                test_by_dataset.get(dataset_id, []), all_models, dataset_id, "test"
            )
            all_dataset_data[dataset_id] = (train_data, test_data)

        # Step 3: Perform global clustering across all datasets
        logger.info("Performing global clustering across all datasets...")
        dataset_cluster_ids = self._assign_clusters_globally(all_dataset_data)

        # Step 4: Add cluster_id to each data item and write files
        train_files = []
        test_files = []

        for dataset_id in sorted(all_dataset_data.keys()):
            logger.info(f"Writing files for dataset: {dataset_id}")

            train_data, test_data = all_dataset_data[dataset_id]
            train_cluster_ids, test_cluster_ids = dataset_cluster_ids[dataset_id]

            # Add cluster_id to train data
            for i, item in enumerate(train_data):
                item["cluster_id"] = train_cluster_ids[i]

            # Add cluster_id to test data
            for i, item in enumerate(test_data):
                item["cluster_id"] = test_cluster_ids[i]

            # Write train file
            if train_data:
                train_file = output_path / f"{dataset_id}_train.json"
                with open(train_file, 'w', encoding='utf-8') as f:
                    json.dump(train_data, f, indent=2, ensure_ascii=False)
                train_files.append(str(train_file))
                logger.info(f"Wrote {len(train_data)} train prompts to {train_file}")

            # Write test file
            if test_data:
                test_file = output_path / f"{dataset_id}_test.json"
                with open(test_file, 'w', encoding='utf-8') as f:
                    json.dump(test_data, f, indent=2, ensure_ascii=False)
                test_files.append(str(test_file))
                logger.info(f"Wrote {len(test_data)} test prompts to {test_file}")

        logger.info("RouterDC conversion complete!")

        return {
            'train': train_files,
            'test': test_files
        }

    def _initialize_embedding_generator(self):
        """Initialize the EmbeddingGenerator from configuration file"""
        try:
            # Load embedding configuration
            with open(self.embedding_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Get model config and cache config
            model_config = config.get('embedding_model', {})
            cache_config = config.get('cache')

            # Read API key from environment if needed
            import os
            api_key = model_config.get('api_key', '')
            if api_key and api_key.isupper() and '_' in api_key:
                # Looks like an environment variable name
                api_key = os.getenv(api_key, api_key)
                model_config['api_key'] = api_key

            # Create embedding generator
            self.embedding_generator = create_generator(model_config, cache_config)
            logger.info(f"Initialized EmbeddingGenerator with model {model_config.get('api_model_name')}")

        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {e}")
            raise

    def _convert_records_to_format(
        self,
        records: List[BaselineRecord],
        all_models: List[str],
        dataset_id: str,
        split_name: str
    ) -> List[Dict]:
        """
        Convert records to RouterDC format (without cluster_id).

        Groups records by prompt and fills missing model scores.

        Args:
            records: List of baseline records (already split)
            all_models: List of all expected model names
            dataset_id: Dataset identifier (for logging)
            split_name: "train" or "test" (for logging)

        Returns:
            List of data items in RouterDC format (without cluster_id)
        """
        if not records:
            return []

        # Group by prompt
        prompt_to_records = defaultdict(list)
        for record in records:
            prompt_to_records[record.prompt].append(record)

        # Statistics for filled values
        from collections import Counter
        filled_counter = Counter()

        # Convert to format
        data = []
        for prompt, prompt_records in prompt_to_records.items():
            scores = {r.model_name: r.score for r in prompt_records}
            base_index = min(r.record_index for r in prompt_records)

            # Fill missing models with 0.0
            filled = fill_missing_models_scores(scores, all_models, fill_value=0.0)
            for model in filled:
                filled_counter[model] = filled_counter.get(model, 0) + 1

            data.append({
                "index": base_index,
                "question": prompt,
                "scores": scores
            })

        # Log statistics
        if filled_counter:
            log_filled_statistics(
                filled_counter,
                prefix=f"Dataset {dataset_id} ({split_name}): ",
                top_n=999
            )

        return data

    def _assign_clusters_globally(
        self,
        all_dataset_data: Dict[str, Tuple[List[Dict], List[Dict]]]
    ) -> Dict[str, Tuple[List[int], List[int]]]:
        """
        Assign cluster IDs globally across all datasets.

        This method:
        1. Collects all train prompts from all datasets
        2. Generates embeddings and performs K-means clustering on all train data
        3. Predicts cluster IDs for all test prompts using the trained model

        Args:
            all_dataset_data: Dict mapping dataset_id to (train_data, test_data) tuples
                              Each data item is a dict with "index", "question", "scores"

        Returns:
            Dict mapping dataset_id to (train_cluster_ids, test_cluster_ids) tuples
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import Normalizer
            import numpy as np
        except ImportError:
            logger.error("scikit-learn not installed. Please install it: pip install scikit-learn")
            raise

        # Step 1: Collect all train prompts from all datasets
        all_train_prompts = []
        dataset_train_ranges = {}  # Maps dataset_id to (start_idx, end_idx) in all_train_prompts

        current_idx = 0
        for dataset_id in sorted(all_dataset_data.keys()):
            train_data, _ = all_dataset_data[dataset_id]
            train_prompts = [item["question"] for item in train_data]

            dataset_train_ranges[dataset_id] = (current_idx, current_idx + len(train_prompts))
            all_train_prompts.extend(train_prompts)
            current_idx += len(train_prompts)

        logger.info(f"Collected {len(all_train_prompts)} total train prompts from {len(all_dataset_data)} datasets")

        # Step 2: Generate embeddings for all train prompts
        logger.info("Generating embeddings for all train prompts...")
        train_embeddings = []
        for prompt in all_train_prompts:
            try:
                result = self.embedding_generator.generate_embedding(prompt)
                if result.embeddings:
                    train_embeddings.append(result.embeddings)
                else:
                    logger.warning(f"Empty embedding for prompt: {prompt[:50]}...")
                    train_embeddings.append([0.0] * 1024)  # Default dimension
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                train_embeddings.append([0.0] * 1024)

        train_embeddings = np.array(train_embeddings, dtype=np.float32)
        logger.debug(f"Generated train embeddings shape: {train_embeddings.shape}")

        # Step 3: Normalize and cluster all train embeddings
        logger.info("Fitting L2 normalizer on all train embeddings...")
        normalizer = Normalizer(norm="l2")
        train_embeddings_normalized = normalizer.fit_transform(train_embeddings)

        # Adjust n_clusters if needed
        n_clusters = min(self.n_clusters, len(all_train_prompts))
        logger.info(f"Fitting K-means with {n_clusters} clusters on all normalized train data...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        all_train_cluster_ids = kmeans.fit_predict(train_embeddings_normalized)

        logger.info(f"Assigned {len(set(all_train_cluster_ids))} unique clusters to {len(all_train_prompts)} train prompts")

        # Step 4: Split train cluster IDs by dataset
        dataset_train_clusters = {}
        for dataset_id, (start_idx, end_idx) in dataset_train_ranges.items():
            dataset_train_clusters[dataset_id] = all_train_cluster_ids[start_idx:end_idx].tolist()

        # Step 5: Collect all test prompts from all datasets
        all_test_prompts = []
        dataset_test_ranges = {}  # Maps dataset_id to (start_idx, end_idx) in all_test_prompts

        current_idx = 0
        for dataset_id in sorted(all_dataset_data.keys()):
            _, test_data = all_dataset_data[dataset_id]
            test_prompts = [item["question"] for item in test_data]

            dataset_test_ranges[dataset_id] = (current_idx, current_idx + len(test_prompts))
            all_test_prompts.extend(test_prompts)
            current_idx += len(test_prompts)

        logger.info(f"Collected {len(all_test_prompts)} total test prompts from {len(all_dataset_data)} datasets")

        # Step 6: Generate embeddings for all test prompts
        if all_test_prompts:
            logger.info("Generating embeddings for all test prompts...")
            test_embeddings = []
            for prompt in all_test_prompts:
                try:
                    result = self.embedding_generator.generate_embedding(prompt)
                    if result.embeddings:
                        test_embeddings.append(result.embeddings)
                    else:
                        test_embeddings.append([0.0] * 1024)
                except Exception as e:
                    logger.error(f"Failed to generate embedding: {e}")
                    test_embeddings.append([0.0] * 1024)

            test_embeddings = np.array(test_embeddings, dtype=np.float32)
            logger.debug(f"Generated test embeddings shape: {test_embeddings.shape}")

            # Step 7: Normalize and predict clusters for all test prompts
            logger.info("Normalizing and predicting clusters for all test prompts...")
            test_embeddings_normalized = normalizer.transform(test_embeddings)
            all_test_cluster_ids = kmeans.predict(test_embeddings_normalized)

            logger.info(f"Assigned {len(set(all_test_cluster_ids))} unique clusters to {len(all_test_prompts)} test prompts")

            # Step 8: Split test cluster IDs by dataset
            dataset_test_clusters = {}
            for dataset_id, (start_idx, end_idx) in dataset_test_ranges.items():
                dataset_test_clusters[dataset_id] = all_test_cluster_ids[start_idx:end_idx].tolist()
        else:
            dataset_test_clusters = {dataset_id: [] for dataset_id in all_dataset_data.keys()}

        # Step 9: Combine results
        result = {}
        for dataset_id in all_dataset_data.keys():
            result[dataset_id] = (
                dataset_train_clusters.get(dataset_id, []),
                dataset_test_clusters.get(dataset_id, [])
            )

        return result


def main():
    """Main entry point for RouterDC adaptor."""
    parser = argparse.ArgumentParser(
        description='Convert LLMRouterBench baseline data to RouterDC format'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/baseline_config.yaml',
        help='Path to baseline configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/test splitting and clustering'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (0.0-1.0)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=8,
        help='Number of clusters for K-means'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='baselines/RouterDC/data',
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--embedding-config',
        type=str,
        default='config/embedding_config.yaml',
        help='Path to embedding model configuration file'
    )
    parser.add_argument(
        '--ood-datasets',
        type=str,
        default='',
        help='Comma-separated list of dataset IDs to treat as OOD (all go to test). Example: brainteaser,dailydialog'
    )

    args = parser.parse_args()
    
    # Parse OOD datasets
    ood_datasets = []
    if args.ood_datasets:
        ood_datasets = [d.strip() for d in args.ood_datasets.split(',') if d.strip()]

    # Create adaptor and convert
    adaptor = RouterDCAdaptor(
        config_path=args.config,
        random_seed=args.seed,
        train_ratio=args.split_ratio,
        n_clusters=args.n_clusters,
        embedding_config_path=args.embedding_config,
        ood_datasets=ood_datasets
    )

    output_files = adaptor.convert(output_dir=args.output_dir)

    logger.info("Generated files:")
    logger.info(f"  Train files ({len(output_files['train'])}):")
    for path in output_files['train']:
        logger.info(f"    {path}")
    logger.info(f"  Test files ({len(output_files['test'])}):")
    for path in output_files['test']:
        logger.info(f"    {path}")


if __name__ == '__main__':
    main()
