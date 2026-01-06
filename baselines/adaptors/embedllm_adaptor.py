"""
EmbedLLM data adaptor for LLMRouterBench.

Converts baseline benchmark results to EmbedLLM format (CSV files).
"""

import argparse
import csv
import os
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from .common import get_unique_models, fill_missing_models_scores, log_filled_statistics
from generators.factory import create_generator
from generators.generator import EmbeddingGenerator


class EmbedLLMAdaptor:
    """
    Adaptor for converting baseline data to EmbedLLM format.

    EmbedLLM requires three CSV files:
    1. question_order_ours.csv: prompt_id, prompt
    2. train_ours.csv: prompt_id, model_id, category_id, label, model_name, category, prompt
    3. test_ours.csv: (same format as train)

    The data is split by dataset, ensuring each dataset has representation in
    both train and test sets.
    """

    def __init__(self, config_path: str, random_seed: int = 42, train_ratio: float = 0.8, ood_datasets: Optional[List[str]] = None):
        """
        Initialize EmbedLLM adaptor.

        Args:
            config_path: Path to baseline configuration YAML file
            random_seed: Random seed for train/test splitting
            train_ratio: Proportion of data for training (0.0-1.0)
            ood_datasets: Optional list of dataset IDs to treat as OOD (all go to test)
        """
        self.config_path = config_path
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.ood_datasets = ood_datasets or []
        self.loader = BaselineDataLoader(config_path=config_path)

        logger.info(f"Initialized EmbedLLM adaptor with seed={random_seed}, ratio={train_ratio}")
        if self.ood_datasets:
            logger.info(f"OOD datasets: {self.ood_datasets}")

    def convert(self, output_dir: str = "baselines/EmbedLLM/data") -> Dict[str, str]:
        """
        Convert baseline data to EmbedLLM format.

        Args:
            output_dir: Directory to save output CSV files

        Returns:
            Dictionary mapping output type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all records
        logger.info("Loading baseline records...")
        all_records = self.loader.load_all_records()
        logger.info(f"Loaded {len(all_records)} records")

        # Split by prompt (not by record) to avoid data leakage
        train_records, test_records = self._split_records_by_prompt(all_records)

        logger.info(f"Split: {len(train_records)} train records, {len(test_records)} test records")

        # Get unique models and prompts from ALL data (train + test combined)
        all_models = get_unique_models(all_records)
        all_prompts_set = set()
        for record in all_records:
            all_prompts_set.add(record.prompt)
        all_prompts = sorted(all_prompts_set)

        # Create mappings
        model_to_id = {model: idx for idx, model in enumerate(all_models)}
        prompt_to_id = {prompt: idx for idx, prompt in enumerate(all_prompts)}

        logger.info(f"Found {len(all_models)} unique models, {len(all_prompts)} unique prompts")

        # Create output directory with seed and split ratio
        seed_str = f"seed{self.random_seed}"
        split_str = f"split{self.train_ratio:.1f}"
        output_path = output_path / f"{seed_str}_{split_str}"
        output_path.mkdir(parents=True, exist_ok=True)

        question_order_file = output_path / "question_order_ours.csv"
        train_file = output_path / "train_ours.csv"
        test_file = output_path / "test_ours.csv"

        # Write question_order_ours.csv
        logger.info(f"Writing question order to {question_order_file}")
        with open(question_order_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt_id', 'prompt'])
            for prompt_id, prompt in enumerate(all_prompts):
                writer.writerow([prompt_id, prompt])

        # Write train_ours.csv
        logger.info(f"Writing train data to {train_file}")
        self._write_data_csv(train_file, train_records, prompt_to_id, model_to_id)

        # Write test_ours.csv
        logger.info(f"Writing test data to {test_file}")
        self._write_data_csv(test_file, test_records, prompt_to_id, model_to_id)

        logger.info("EmbedLLM conversion complete!")

        return {
            'question_order': str(question_order_file),
            'train': str(train_file),
            'test': str(test_file)
        }

    def _split_records_by_prompt(
        self,
        all_records: List[BaselineRecord]
    ) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
        """
        Split records into train/test by dataset, then by prompts within each dataset.

        This ensures that:
        1. Each dataset is split independently
        2. All model evaluations for the same prompt stay together
        3. Prevents data leakage across prompts

        Uses BaselineDataLoader's unified splitting logic.

        Args:
            all_records: All baseline records

        Returns:
            Tuple of (train_records, test_records)
        """
        # Use BaselineDataLoader's unified splitting logic
        return self.loader.split_by_dataset_then_prompt(
            all_records,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            ood_datasets=self.ood_datasets
        )

    def _write_data_csv(
        self,
        file_path: Path,
        records: List[BaselineRecord],
        prompt_to_id: Dict[str, int],
        model_to_id: Dict[str, int]
    ):
        """
        Write data CSV file (train or test) with missing data filled as 0.0.

        Args:
            file_path: Output file path
            records: List of baseline records
            prompt_to_id: Mapping from prompt to prompt_id
            model_to_id: Mapping from model name to model_id
        """
        from collections import Counter, defaultdict

        # Group records by prompt
        prompt_to_records = defaultdict(list)
        for record in records:
            prompt_to_records[record.prompt].append(record)

        # Statistics for filled values
        filled_counter = Counter()
        total_records = 0

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'prompt_id', 'model_id', 'category_id',
                'label', 'model_name', 'category', 'prompt'
            ])

            # Write records for all prompts
            for prompt in sorted(prompt_to_records.keys()):
                prompt_id = prompt_to_id[prompt]

                # Build scores dict for this prompt
                prompt_records = prompt_to_records[prompt]
                scores = {r.model_name: r.score for r in prompt_records}

                # Use first record for category information
                sample_record = prompt_records[0]

                # Fill missing models with 0.0 (unified logic)
                all_models = list(model_to_id.keys())
                filled = fill_missing_models_scores(scores, all_models, fill_value=0.0)
                for model in filled:
                    filled_counter[model] = filled_counter.get(model, 0) + 1

                # Write all model scores for this prompt
                for model_name in sorted(model_to_id.keys()):
                    model_id = model_to_id[model_name]
                    label = scores[model_name]  # Now guaranteed to exist after filling

                    writer.writerow([
                        prompt_id,
                        model_id,
                        sample_record.dataset_id,  # category_id
                        label,
                        model_name,
                        sample_record.dataset_id,  # category
                        prompt
                    ])
                    total_records += 1

        # Log statistics
        total_filled = sum(filled_counter.values())
        original_records = len(records)

        logger.info(f"Wrote {total_records} records to {file_path}")
        logger.info(f"  Original: {original_records} records")
        logger.info(f"  Filled: {total_filled} records ({total_filled/total_records*100:.1f}%)")

        if filled_counter:
            logger.info(f"  Filled by model:")
            log_filled_statistics(filled_counter, prefix="    ", top_n=999)  # Show all models

    def generate_question_embeddings(
        self,
        question_order_csv_path: str,
        embedding_config_path: str = "config/embedding_config.yaml",
        output_path: str = None
    ) -> torch.Tensor:
        """
        Generate embeddings for questions in the question_order CSV file.

        Args:
            question_order_csv_path: Path to the question_order CSV file (with 'prompt' column)
            embedding_config_path: Path to the embedding model configuration YAML file
            output_path: Optional path to save the output tensor (.pth file)

        Returns:
            torch.Tensor: Tensor of shape (num_questions, embedding_dim) containing question embeddings
        """
        logger.info(f"Loading questions from {question_order_csv_path}")

        # Load questions from CSV
        df = pd.read_csv(question_order_csv_path)
        questions = df['prompt'].tolist()
        logger.info(f"Loaded {len(questions)} questions")

        # Load embedding configuration
        logger.info(f"Loading embedding config from {embedding_config_path}")
        with open(embedding_config_path, 'r', encoding='utf-8') as f:
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
        logger.info(f"Initializing EmbeddingGenerator with model {model_config.get('api_model_name')}")
        embedding_generator = create_generator(model_config, cache_config)

        # Generate embeddings with tqdm progress bar
        embeddings = []
        for question in tqdm(questions, desc="Generating embeddings"):
            try:
                result = embedding_generator.generate_embedding(question)
                if result.embeddings:
                    embeddings.append(result.embeddings)
                else:
                    logger.warning(f"Empty embedding for question: {question[:50]}...")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 1024)  # Will be resized to match actual dimension
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 1024)  # Will be resized to match actual dimension

        # Convert to numpy array first to handle different dimensions
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Convert to PyTorch tensor
        embedding_tensor = torch.from_numpy(embeddings_array)

        logger.info(f"Generated embeddings tensor of shape: {embedding_tensor.shape}")

        # Save the tensor if output path is provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding_tensor, output_file)
            logger.info(f"Saved embeddings to: {output_file}")

        return embedding_tensor


def main():
    """Main entry point for EmbedLLM adaptor."""
    parser = argparse.ArgumentParser(
        description='Convert LLMRouterBench baseline data to EmbedLLM format'
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
        help='Random seed for train/test splitting'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (0.0-1.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='baselines/EmbedLLM/data',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--embedding-config',
        type=str,
        default='config/embedding_config.yaml',
        help='Path to embedding model configuration file'
    )
    parser.add_argument(
        '--embedding-output',
        type=str,
        default=None,
        help='Path to save the embeddings tensor (.pth file). If not specified, will be saved in output directory'
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
    adaptor = EmbedLLMAdaptor(
        config_path=args.config,
        random_seed=args.seed,
        train_ratio=args.split_ratio,
        ood_datasets=ood_datasets
    )

    output_files = adaptor.convert(output_dir=args.output_dir)

    logger.info("Generated files:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")

    # Generate question embeddings automatically
    logger.info("Generating question embeddings...")
    question_order_file = output_files['question_order']

    # Determine output path for embeddings
    if args.embedding_output:
        embedding_output_path = args.embedding_output
    else:
        # Save to same directory as question_order file
        question_order_path = Path(question_order_file)
        embedding_output_path = question_order_path.parent / "question_embeddings.pth"

    # Generate and save embeddings
    adaptor.generate_question_embeddings(
        question_order_csv_path=question_order_file,
        embedding_config_path=args.embedding_config,
        output_path=str(embedding_output_path)
    )

    logger.info("EmbedLLM data preparation complete!")


if __name__ == '__main__':
    main()
