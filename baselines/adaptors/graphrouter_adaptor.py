"""
GraphRouter data adaptor for LLMRouterBench.

Converts baseline benchmark results to GraphRouter format (CSV + embeddings).
"""

import argparse
import json
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from loguru import logger
from tqdm import tqdm

import numpy as np
import pandas as pd

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from .common import get_unique_models, log_filled_statistics
from generators.factory import create_generator


def format_embedding_for_graphrouter(embedding: List[float]) -> str:
    """
    Format embedding as GraphRouter expects: [[ val1 val2 val3 ... ]]

    Avoids numpy print truncation by manually constructing the string.

    Args:
        embedding: Embedding vector as list or numpy array

    Returns:
        String in GraphRouter format with space-separated values
    """
    # Ensure it's a flat list
    if isinstance(embedding, np.ndarray):
        embedding = embedding.flatten().tolist()
    elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]

    # Manual string construction - immune to np.set_printoptions
    values_str = ' '.join(str(v) for v in embedding)
    return f'[[{values_str}]]'


class GraphRouterAdaptor:
    """
    Adaptor for converting baseline data to GraphRouter format.

    GraphRouter requires:
    1. router_data.csv: 10-column CSV with query, embeddings, scores, costs
    2. llm_description_embedding.pkl: Embeddings of LLM feature descriptions

    Each row in CSV represents a (task_id, query, llm) evaluation result.
    """

    def __init__(
        self,
        baseline_config_path: str = "config/baseline_config.yaml",
        graphrouter_config_path: str = "baselines/adaptors/graphrouter_config.yaml",
        embedding_config_path: str = "config/embedding_config.yaml",
        random_seed: int = 42
    ):
        """
        Initialize GraphRouter adaptor.

        Args:
            baseline_config_path: Path to baseline configuration YAML file
            graphrouter_config_path: Path to GraphRouter configuration YAML file
            embedding_config_path: Path to embedding model configuration YAML file
            random_seed: Random seed for reproducibility
        """
        self.baseline_config_path = baseline_config_path
        self.graphrouter_config_path = graphrouter_config_path
        self.embedding_config_path = embedding_config_path
        self.random_seed = random_seed

        # Load configurations
        self.baseline_loader = BaselineDataLoader(config_path=baseline_config_path)
        self.graphrouter_config = self._load_graphrouter_config()
        self.embedding_generator = self._initialize_embedding_generator()

        logger.info(f"Initialized GraphRouter adaptor with seed={random_seed}")

    def _load_graphrouter_config(self) -> Dict:
        """Load GraphRouter configuration file."""
        with open(self.graphrouter_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('graphrouter', {})

    def _initialize_embedding_generator(self):
        """Initialize the EmbeddingGenerator from configuration file."""
        try:
            import os

            # Load embedding configuration
            with open(self.embedding_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Get model config and cache config
            model_config = config.get('embedding_model', {})
            cache_config = config.get('cache')

            # Read API key from environment if needed
            api_key = model_config.get('api_key', '')
            if api_key and api_key.isupper() and '_' in api_key:
                # Looks like an environment variable name
                api_key = os.getenv(api_key, api_key)
                model_config['api_key'] = api_key

            # Create embedding generator
            embedding_gen = create_generator(model_config, cache_config)
            logger.info(f"Initialized EmbeddingGenerator with model {model_config.get('api_model_name')}")
            return embedding_gen

        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {e}")
            raise

    def _get_llm_description(self, model_name: str) -> Tuple[str, float]:
        """
        Get LLM feature description and price.

        Args:
            model_name: Model identifier

        Returns:
            Tuple of (feature_description, price)
        """
        llm_descriptions = self.graphrouter_config.get('llm_descriptions', {})

        if model_name in llm_descriptions:
            desc = llm_descriptions[model_name]
            return desc['feature'], desc.get('input_price', 0.2)
        else:
            raise KeyError(
                f"Model '{model_name}' not found in graphrouter_config.yaml llm_descriptions. "
                f"Available models: {list(llm_descriptions.keys())}"
            )

    def _get_task_description(self, dataset_id: str) -> Tuple[str, str]:
        """
        Get task description and metric for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Tuple of (description, metric)
        """
        task_descriptions = self.graphrouter_config.get('task_descriptions', {})

        if dataset_id in task_descriptions:
            desc = task_descriptions[dataset_id]
            return desc['description'], desc.get('metric', 'f1_score')
        else:
            raise KeyError(
                f"Dataset '{dataset_id}' not found in graphrouter_config.yaml task_descriptions. "
                f"Available datasets: {list(task_descriptions.keys())}"
            )

    def convert(self, output_dir: str = "baselines/GraphRouter/data") -> Dict[str, str]:
        """
        Convert baseline data to GraphRouter format.

        Args:
            output_dir: Directory to save output files

        Returns:
            Dictionary mapping output type to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all records
        logger.info("Loading baseline records...")
        all_records = self.baseline_loader.load_all_records()
        logger.info(f"Loaded {len(all_records)} records")

        # Get unique models
        all_models = get_unique_models(all_records)
        logger.info(f"Found {len(all_models)} unique models")

        # Generate router_data.csv and router_data.parquet
        csv_file = output_path / self.graphrouter_config['output']['router_data_csv']
        parquet_file = output_path / self.graphrouter_config['output']['router_data_csv'].replace('.csv', '.parquet')
        logger.info(f"Generating router_data files at {csv_file}")
        self._generate_router_data_csv(all_records, all_models, csv_file)

        # Generate llm_description_embedding.pkl
        pkl_file = output_path / self.graphrouter_config['output']['llm_embedding_pkl']
        logger.info(f"Generating LLM embeddings at {pkl_file}")
        self._generate_llm_embeddings(all_models, pkl_file)

        # Generate LLM_Descriptions.json
        json_file = output_path / self.graphrouter_config['output']['llm_descriptions_json']
        logger.info(f"Generating LLM descriptions JSON at {json_file}")
        self._generate_llm_descriptions_json(all_models, json_file)

        logger.info("GraphRouter conversion complete!")

        return {
            'router_data_csv': str(csv_file),
            'router_data_parquet': str(parquet_file),
            'llm_embedding_pkl': str(pkl_file),
            'llm_descriptions_json': str(json_file)
        }

    def _generate_router_data_csv(
        self,
        records: List[BaselineRecord],
        all_models: List[str],
        output_path: Path
    ):
        """
        Generate router_data.csv file and router_data.parquet file.

        CSV/Parquet columns:
        - task_id, query, query_embedding, ground_truth, metric, llm, effect,
          cost (normalized by task for training), cost_usd (raw USD for stats),
          task_description, task_description_embedding, record_index

        Splitting/ordering:
        - Reproduce Avengers split logic (seeded per-dataset prompt split)
        - Ensure train queries appear first within each dataset and test queries
          appear after, so GraphRouter's sequential 70/30 split matches Avengers

        Args:
            records: List of baseline records
            all_models: List of all model names
            output_path: Output CSV file path
        """
        logger.info("Preparing data for CSV generation...")

        # Build unique query groups by (dataset_id, prompt)
        query_groups = defaultdict(list)
        for record in records:
            key = (record.dataset_id, record.prompt)
            query_groups[key].append(record)

        # Derive Avengers-like split by dataset and prompt (no external files)
        logger.info("Deriving Avengers-style split (per-dataset, per-prompt, seeded shuffle)...")
        train_records, test_records = self.baseline_loader.split_by_dataset_then_prompt(
            records,
            train_ratio=self.graphrouter_config.get('split', {}).get('train_ratio', 0.7),
            random_seed=self.random_seed,
            ood_datasets=None
        )
        test_prompt_set = set((r.dataset_id, r.prompt) for r in test_records)

        # Prepare CSV rows
        csv_rows = []

        # Cache for embeddings
        query_embedding_cache = {}
        task_embedding_cache = {}

        logger.info(f"Processing {len(query_groups)} unique queries...")

        for (dataset_id, prompt), group_records in tqdm(query_groups.items(), desc="Processing queries"):
            # Get task description and metric
            task_description, metric = self._get_task_description(dataset_id)

            # Generate query embedding (cache to avoid regenerating for same query)
            if prompt not in query_embedding_cache:
                try:
                    result = self.embedding_generator.generate_embedding(prompt)
                    if result.embeddings:
                        query_embedding_cache[prompt] = result.embeddings
                    else:
                        logger.warning(f"Empty embedding for query: {prompt[:50]}...")
                        query_embedding_cache[prompt] = [0.0] * 768  # Default dimension
                except Exception as e:
                    logger.error(f"Error generating query embedding: {e}")
                    query_embedding_cache[prompt] = [0.0] * 768

            query_embedding = query_embedding_cache[prompt]

            # Generate task description embedding (cache by dataset)
            if dataset_id not in task_embedding_cache:
                try:
                    result = self.embedding_generator.generate_embedding(task_description)
                    if result.embeddings:
                        task_embedding_cache[dataset_id] = result.embeddings
                    else:
                        logger.warning(f"Empty embedding for task: {dataset_id}")
                        task_embedding_cache[dataset_id] = [0.0] * 768
                except Exception as e:
                    logger.error(f"Error generating task embedding: {e}")
                    task_embedding_cache[dataset_id] = [0.0] * 768

            task_embedding = task_embedding_cache[dataset_id]

            # Ground truth and a stable index for ordering
            ground_truth = group_records[0].ground_truth
            record_index = min(r.record_index for r in group_records)

            # Create a row for each model evaluation
            model_scores = {r.model_name: r for r in group_records}

            for model_name in all_models:
                if model_name in model_scores:
                    record = model_scores[model_name]
                    effect = record.score
                    cost_usd = record.cost
                else:
                    # Model didn't evaluate this query
                    effect = 0.0
                    cost_usd = 0.0

                # Create CSV row; 'cost' will be normalized later per task
                csv_rows.append({
                    'task_id': dataset_id,
                    'query': prompt,
                    'record_index': record_index,
                    'query_embedding': format_embedding_for_graphrouter(query_embedding),
                    'ground_truth': ground_truth,
                    'metric': metric,
                    'llm': model_name,
                    'effect': effect,
                    'cost': cost_usd,        # temp as USD, will keep copy then normalize this column
                    'task_description': task_description,
                    'task_description_embedding': format_embedding_for_graphrouter(task_embedding)
                })

        # Convert to DataFrame
        logger.info(f"Creating DataFrame with {len(csv_rows)} rows...")
        df = pd.DataFrame(csv_rows)

        # Preserve USD cost and create normalized training cost per task
        df['cost_usd'] = df['cost']
        logger.info("Creating normalized cost column for training (per task_id min-max)...")
        df = self._normalize_costs_by_task(df)  # modifies 'cost' in-place

        # Tag test queries (for ordering only) using the derived prompt split
        test_keys = set(test_prompt_set)
        df['is_test'] = df.apply(lambda r: (r['task_id'], r['query']) in test_keys, axis=1)

        # Order rows: by task_id, then train before test, then by record_index to keep queries contiguous
        logger.info("Reordering rows so that train queries appear before test queries within each task...")
        df.sort_values(by=['task_id', 'is_test', 'record_index'], kind='mergesort', inplace=True)

        # Drop helper flag from final output
        df.drop(columns=['is_test'], inplace=True)

        # Write to CSV
        logger.info(f"Writing CSV to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully wrote {len(df)} rows to {output_path}")

        # Also write to Parquet format for better performance
        parquet_path = output_path.parent / output_path.name.replace('.csv', '.parquet')
        logger.info(f"Writing Parquet to {parquet_path}...")
        df.to_parquet(parquet_path, compression='snappy', index=False)
        logger.info(f"Successfully wrote {len(df)} rows to {parquet_path} (Parquet format)")

    def _normalize_costs_by_task(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize costs by task_id using min-max normalization.

        Args:
            df: DataFrame with 'task_id' and 'cost' columns

        Returns:
            DataFrame with normalized costs
        """
        def normalize_group(group):
            costs = group['cost']
            min_cost = costs.min()
            max_cost = costs.max()

            if max_cost > min_cost:
                group['cost'] = (costs - min_cost) / (max_cost - min_cost)
            else:
                # All costs are the same
                group['cost'] = 0.0

            return group

        df = df.groupby('task_id', group_keys=False).apply(normalize_group)
        return df

    def _generate_llm_embeddings(
        self,
        all_models: List[str],
        output_path: Path
    ):
        """
        Generate embeddings for LLM feature descriptions.

        Args:
            all_models: List of model names
            output_path: Output pickle file path
        """
        logger.info(f"Generating embeddings for {len(all_models)} LLM descriptions...")

        llm_features = []
        for model_name in tqdm(all_models, desc="Generating LLM embeddings"):
            feature_desc, _ = self._get_llm_description(model_name)
            llm_features.append(feature_desc)

        # Generate embeddings
        embeddings = []
        for feature in llm_features:
            try:
                result = self.embedding_generator.generate_embedding(feature)
                if result.embeddings:
                    embeddings.append(result.embeddings)
                else:
                    logger.warning(f"Empty embedding for feature: {feature[:50]}...")
                    embeddings.append([0.0] * 768)
            except Exception as e:
                logger.error(f"Error generating LLM embedding: {e}")
                embeddings.append([0.0] * 768)

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Generated embeddings array of shape: {embeddings_array.shape}")

        # Save to pickle file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_array, f)

        logger.info(f"Saved LLM embeddings to {output_path}")

    def _generate_llm_descriptions_json(
        self,
        all_models: List[str],
        output_path: Path
    ):
        """
        Generate LLM_Descriptions.json file for GraphRouter training.

        Args:
            all_models: List of model names (sorted alphabetically)
            output_path: Output JSON file path
        """
        logger.info(f"Generating LLM descriptions JSON for {len(all_models)} models...")

        # Build LLM descriptions dictionary in the same order as all_models
        llm_desc_dict = {}
        for model_name in all_models:
            feature_desc, _ = self._get_llm_description(model_name)
            llm_desc_dict[model_name] = {
                "feature": feature_desc
            }

        # Save to JSON file (Python 3.7+ preserves insertion order)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(llm_desc_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved LLM descriptions to {output_path}")


def main():
    """Main entry point for GraphRouter adaptor."""
    parser = argparse.ArgumentParser(
        description='Convert LLMRouterBench baseline data to GraphRouter format'
    )
    parser.add_argument(
        '--baseline-config',
        type=str,
        default='config/baseline_config.yaml',
        help='Path to baseline configuration file'
    )
    parser.add_argument(
        '--graphrouter-config',
        type=str,
        default='baselines/adaptors/graphrouter_config.yaml',
        help='Path to GraphRouter configuration file (LLM and task descriptions)'
    )
    parser.add_argument(
        '--embedding-config',
        type=str,
        default='config/embedding_config.yaml',
        help='Path to embedding model configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='baselines/GraphRouter/data',
        help='Output directory for CSV and embedding files'
    )

    args = parser.parse_args()

    # Create adaptor and convert
    adaptor = GraphRouterAdaptor(
        baseline_config_path=args.baseline_config,
        graphrouter_config_path=args.graphrouter_config,
        embedding_config_path=args.embedding_config,
        random_seed=args.seed
    )

    output_files = adaptor.convert(output_dir=args.output_dir)

    logger.info("Generated files:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")

    logger.info("\nTo use these files with GraphRouter:")
    logger.info(f'  router_data_path (CSV): "{output_files["router_data_csv"]}"')
    logger.info(f'  router_data_path (Parquet - Recommended): "{output_files["router_data_parquet"]}"')
    logger.info(f'  llm_description_path: "{output_files["llm_descriptions_json"]}"')
    logger.info(f'  llm_embedding_path: "{output_files["llm_embedding_pkl"]}"')


if __name__ == '__main__':
    main()
