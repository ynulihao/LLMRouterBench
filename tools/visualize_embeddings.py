#!/usr/bin/env python3
"""
Embedding Visualization Tool for LLMRouterBench

This script visualizes question embeddings from benchmark results, colored by dataset.
It loads results from the results/bench directory, generates embeddings using the
EmbeddingGenerator, and creates a 2D visualization using dimensionality reduction.

Usage:
    python -m tools.visualize_embeddings
    python -m tools.visualize_embeddings --max-samples 100 --method pca
    python -m tools.visualize_embeddings --method umap
    python -m tools.visualize_embeddings --config config/embedding_config.yaml --output viz.png
"""

import sys
import os
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.factory import create_generator
from baselines import BaselineDataLoader


def load_benchmark_results(baseline_config: Optional[Dict] = None, max_samples_per_dataset: Optional[int] = None) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Load unique questions from benchmark results using BaselineDataLoader with deduplication.

    This function loads all benchmark results and deduplicates by prompt to ensure
    each unique question is only included once, even if it appears in multiple model results.

    Args:
        baseline_config: Optional configuration dictionary for BaselineDataLoader.
                        If None, uses default configuration.
        max_samples_per_dataset: Optional maximum number of unique samples per dataset.
                                If None, loads all available samples.

    Returns:
        Tuple of (questions, dataset_labels, dataset_counts) where:
        - questions: List of unique question strings (prompts)
        - dataset_labels: List of dataset names corresponding to each question
        - dataset_counts: Dictionary mapping dataset names to their question counts
    """
    # Initialize BaselineDataLoader
    if baseline_config is None:
        baseline_config = {
            'results_dir': 'results/bench',
            'filters': {
                'skip_demo': True,  # Skip demo results
            },
            'output': {
                'use_cache': False,  # Disable caching for visualization
            }
        }

    loader = BaselineDataLoader(config=baseline_config)
    logger.info("Initialized BaselineDataLoader for loading benchmark results")

    # Dictionary to store unique prompts and their datasets
    # Key: prompt (question text), Value: dataset_id
    prompt_to_dataset = {}

    # Counter for dataset-specific limits
    dataset_sample_counts = defaultdict(int)

    # Statistics
    total_records = 0
    skipped_duplicates = 0

    logger.info("Loading records and deduplicating by prompt...")

    # Iterate over all records
    for record in tqdm(loader.load_records_iter(), desc="Loading and deduplicating records"):
        total_records += 1

        # Check dataset-specific sample limit
        if max_samples_per_dataset is not None:
            if dataset_sample_counts[record.dataset_id] >= max_samples_per_dataset:
                continue

        # Check if we've seen this prompt before
        if record.prompt in prompt_to_dataset:
            skipped_duplicates += 1
            continue

        # Add unique prompt
        prompt_to_dataset[record.prompt] = record.dataset_id
        dataset_sample_counts[record.dataset_id] += 1

    # Convert to lists
    questions = list(prompt_to_dataset.keys())
    dataset_labels = [prompt_to_dataset[q] for q in questions]
    dataset_counts = dict(dataset_sample_counts)

    # Log statistics
    unique_questions = len(questions)
    dedup_ratio = (skipped_duplicates / total_records * 100) if total_records > 0 else 0

    logger.info(f"Total records processed: {total_records}")
    logger.info(f"Unique questions after deduplication: {unique_questions}")
    logger.info(f"Duplicate records skipped: {skipped_duplicates} ({dedup_ratio:.1f}%)")
    logger.info(f"Datasets: {len(dataset_counts)}")

    # Log per-dataset statistics
    logger.info("Per-dataset unique question counts:")
    for dataset_name in sorted(dataset_counts.keys()):
        count = dataset_counts[dataset_name]
        logger.info(f"  {dataset_name}: {count} unique questions")

    return questions, dataset_labels, dataset_counts


def generate_embeddings(questions: List[str], embedding_config_path: Path, max_workers: int = 8) -> np.ndarray:
    """
    Generate embeddings for all questions using EmbeddingGenerator with concurrent processing.

    Args:
        questions: List of question strings
        embedding_config_path: Path to embedding configuration YAML file
        max_workers: Number of concurrent workers (default: 8)

    Returns:
        Numpy array of shape (n_questions, embedding_dim)
    """
    # Load embedding configuration
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
    logger.info(f"Initializing EmbeddingGenerator with model {model_config.get('api_model_name')}...")
    embedding_generator = create_generator(model_config, cache_config)

    # Helper function to generate single embedding
    def _generate_single_embedding(idx_question):
        idx, question = idx_question
        try:
            result = embedding_generator.generate_embedding(question)
            if result.embeddings:
                return (idx, result.embeddings, None)
            else:
                return (idx, None, f"Empty embedding for question: {question[:50]}...")
        except Exception as e:
            return (idx, None, str(e))

    # Generate embeddings with concurrent processing
    logger.info(f"Generating embeddings for {len(questions)} questions using {max_workers} workers...")

    # Initialize results array to maintain order
    embeddings = [None] * len(questions)
    default_dim = 1024  # Default dimension for fallback

    # Create index-question pairs
    indexed_questions = list(enumerate(questions))

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_generate_single_embedding, iq): iq for iq in indexed_questions}

        # Process completed tasks with progress bar
        with tqdm(total=len(questions), desc="Generating embeddings") as pbar:
            for future in as_completed(futures):
                idx, embedding, error = future.result()

                if embedding is not None:
                    embeddings[idx] = embedding
                    # Update default dimension based on first successful embedding
                    if default_dim == 1024 and len(embedding) != 1024:
                        default_dim = len(embedding)
                else:
                    # Use zero vector as fallback
                    if error:
                        logger.warning(f"Index {idx}: {error}")
                    embeddings[idx] = [0.0] * default_dim

                pbar.update(1)

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")

    return embeddings_array


def reduce_dimensions(embeddings: np.ndarray, method: str = 'tsne', random_state: int = 42) -> np.ndarray:
    """
    Reduce embedding dimensions to 2D for visualization.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 2)
    """
    logger.info(f"Reducing dimensions using {method.upper()}...")

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'tsne', 'pca', or 'umap'")

    embeddings_2d = reducer.fit_transform(embeddings)
    logger.info(f"Reduced to 2D shape: {embeddings_2d.shape}")

    return embeddings_2d


def visualize_embeddings(embeddings_2d: np.ndarray,
                        dataset_labels: List[str],
                        dataset_counts: Dict[str, int],
                        output_path: Path,
                        method: str = 'tsne'):
    """
    Create and save a scatter plot of 2D embeddings colored by dataset.

    Args:
        embeddings_2d: Array of shape (n_samples, 2)
        dataset_labels: List of dataset names for each sample
        dataset_counts: Dictionary mapping dataset names to sample counts
        output_path: Path to save the visualization
        method: Dimensionality reduction method used (for title)
    """
    logger.info("Creating visualization...")

    # Get unique datasets and assign colors
    unique_datasets = sorted(set(dataset_labels))
    n_datasets = len(unique_datasets)

    # Use a colormap with enough distinct colors
    if n_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_datasets]
    elif n_datasets <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_datasets]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_datasets))

    dataset_to_color = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each dataset separately to create legend entries
    for dataset in unique_datasets:
        # Get indices for this dataset
        indices = [i for i, label in enumerate(dataset_labels) if label == dataset]

        # Get embeddings for this dataset
        dataset_embeddings = embeddings_2d[indices]

        # Plot with label
        count = dataset_counts[dataset]
        ax.scatter(
            dataset_embeddings[:, 0],
            dataset_embeddings[:, 1],
            c=[dataset_to_color[dataset]],
            label=f'{dataset} (n={count})',
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )

    # Set labels and title
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title(
        f'Question Embeddings Visualization by Dataset ({method.upper()})\n'
        f'Total: {len(embeddings_2d)} questions across {n_datasets} datasets',
        fontsize=14,
        fontweight='bold'
    )

    # Add legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        fontsize=9
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {output_path}")

    # Close figure to free memory
    plt.close(fig)


def main():
    """Main entry point for the visualization tool."""
    parser = argparse.ArgumentParser(
        description='Visualize question embeddings from LLMRouterBench results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (t-SNE)
  python -m tools.visualize_embeddings

  # Use UMAP for better cluster preservation
  python -m tools.visualize_embeddings --method umap

  # Use PCA for faster processing
  python -m tools.visualize_embeddings --method pca

  # Load all available unique samples with 16 concurrent workers
  python -m tools.visualize_embeddings --workers 16 --method umap

  # Limit to 100 unique samples per dataset
  python -m tools.visualize_embeddings --max-samples 100

  # Custom baseline and embedding configs
  python -m tools.visualize_embeddings --baseline-config my_baseline.yaml --config my_embedding.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/embedding_config.yaml',
        help='Path to embedding model configuration file (default: config/embedding_config.yaml)'
    )

    parser.add_argument(
        '--baseline-config',
        type=str,
        default=None,
        help='Path to baseline configuration YAML file (optional, uses default config if not provided)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of unique samples to load per dataset (default: None, loads all available)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['tsne', 'pca', 'umap'],
        default='tsne',
        help='Dimensionality reduction method (default: tsne)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='embedding_visualization.png',
        help='Output path for visualization image (default: embedding_visualization.png)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of concurrent workers for embedding generation (default: 8)'
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    config_path = Path(args.config)
    output_path = Path(args.output)

    # Validate embedding config path
    if not config_path.exists():
        logger.error(f"Embedding config file not found: {config_path}")
        sys.exit(1)

    # Load baseline configuration if provided
    baseline_config = None
    if args.baseline_config:
        baseline_config_path = Path(args.baseline_config)
        if not baseline_config_path.exists():
            logger.error(f"Baseline config file not found: {baseline_config_path}")
            sys.exit(1)

        with open(baseline_config_path, 'r') as f:
            baseline_config = yaml.safe_load(f).get('baseline', {})
        logger.info(f"Loaded baseline configuration from: {baseline_config_path}")

    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

    # Step 1: Load benchmark results
    logger.info("=" * 80)
    logger.info("Step 1: Loading benchmark results with deduplication")
    logger.info("=" * 80)
    questions, dataset_labels, dataset_counts = load_benchmark_results(
        baseline_config=baseline_config,
        max_samples_per_dataset=args.max_samples
    )

    if not questions:
        logger.error("No questions loaded! Check your baseline configuration.")
        sys.exit(1)

    logger.info(f"Total questions loaded: {len(questions)}")
    logger.info(f"Datasets: {len(dataset_counts)}")

    # Step 2: Generate embeddings
    logger.info("=" * 80)
    logger.info("Step 2: Generating embeddings")
    logger.info("=" * 80)
    embeddings = generate_embeddings(questions, config_path, max_workers=args.workers)

    # Step 3: Reduce dimensions
    logger.info("=" * 80)
    logger.info("Step 3: Reducing dimensions")
    logger.info("=" * 80)
    embeddings_2d = reduce_dimensions(embeddings, method=args.method, random_state=args.random_seed)

    # Step 4: Create visualization
    logger.info("=" * 80)
    logger.info("Step 4: Creating visualization")
    logger.info("=" * 80)
    visualize_embeddings(
        embeddings_2d,
        dataset_labels,
        dataset_counts,
        output_path,
        method=args.method
    )

    logger.info("=" * 80)
    logger.info("✅ Visualization complete!")
    logger.info(f"Output saved to: {output_path.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
