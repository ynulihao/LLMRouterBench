#!/usr/bin/env python3
"""
Weight Ablation Runner

Independent script for running cost/performance weight ablation study only.
Provides focused interface for weight balance optimization.
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .weight_ablation import WeightAblation
from ..config import SimpleClusterConfig, setup_logging
from ..simple_cluster_router import SimpleClusterRouter


# Global embedding cache: key=query_string, value=embedding_array
EMBEDDING_CACHE = {}


def prefill_embedding_cache(train_data_path, test_data_path, config, logger):
    """
    Pre-fill the global embedding cache with all unique queries from train and test data.
    This ensures all subsequent experiments only need to read from cache (no writes during parallel execution).

    Args:
        train_data_path: Path to training data file
        test_data_path: Path to test data file
        config: SimpleClusterConfig instance
        logger: Logger instance

    Returns:
        Number of unique queries cached
    """
    logger.info("Starting embedding cache pre-filling...")

    # Collect all unique queries
    unique_queries = set()

    # Load training data
    if train_data_path and Path(train_data_path).exists():
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'query' in item:
                    unique_queries.add(item['query'])

    # Load test data
    if test_data_path and Path(test_data_path).exists():
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'query' in item:
                    unique_queries.add(item['query'])

    logger.info(f"Found {len(unique_queries)} unique queries to cache")

    # Create a temporary router to generate embeddings
    temp_router = SimpleClusterRouter(config)

    # Filter out queries that are already cached
    unique_queries_list = [q for q in unique_queries if q not in EMBEDDING_CACHE]

    if not unique_queries_list:
        logger.info("All queries already cached, skipping pre-fill")
        return len(EMBEDDING_CACHE)

    logger.info(f"Generating embeddings for {len(unique_queries_list)} new queries...")

    # Helper function for concurrent embedding generation
    def generate_single_embedding(query):
        """Generate embedding for a single query."""
        embedding_output = temp_router.embedder.generate_embedding(query)
        embedding = np.array(embedding_output.embeddings, dtype=float)
        return query, embedding

    # Generate embeddings concurrently with progress bar
    max_workers = config.max_workers if hasattr(config, 'max_workers') else 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(generate_single_embedding, query): query
                   for query in unique_queries_list}

        # Process completed tasks with progress bar
        with tqdm(total=len(unique_queries_list), desc="Pre-filling embedding cache") as pbar:
            for future in as_completed(futures):
                query, embedding = future.result()
                EMBEDDING_CACHE[query] = embedding
                pbar.update(1)

    # Set the cache to the router's class variable
    SimpleClusterRouter._global_embedding_cache = EMBEDDING_CACHE

    logger.info(f"Cache pre-filling completed: {len(EMBEDDING_CACHE)} embeddings cached")
    return len(EMBEDDING_CACHE)


def parse_weight_range(weight_range_str):
    """Parse weight range string into tuple."""
    if not weight_range_str:
        return None
    
    parts = weight_range_str.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    else:
        raise ValueError("Weight range must be 'min,max' format")


def main():
    """Main entry point for weight ablation."""
    parser = argparse.ArgumentParser(
        description='Cost/Performance Weight Ablation Study for Balance Cluster Router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic weight ablation with default config
  python run_weight_ablation.py --config config/ablation_weight_config.json

  # Custom weight configuration
  python run_weight_ablation.py --config config/ablation_weight_config.json --weight-range 0,1 --step-size 0.1

  # Override train/test data paths
  python run_weight_ablation.py --config config/ablation_weight_config.json --train-data data/custom_train.jsonl --test-data data/custom_test.jsonl

  # Parallel execution
  python run_weight_ablation.py --config config/ablation_weight_config.json --step-size 0.1 --parallel --quiet
        """
    )

    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file')

    # Optional data path overrides
    parser.add_argument('--train-data', type=str,
                       help='Override train data path from config')
    parser.add_argument('--test-data', type=str,
                       help='Override test data path from config')

    # Weight ablation specific arguments
    parser.add_argument('--weight-range', type=str, default='0,1',
                       help='Performance weight range as "min,max" (default: "0,1")')
    parser.add_argument('--step-size', type=float, default=0.01,
                       help='Step size for performance_weight iteration (default: 0.01)')

    # Output and execution options
    parser.add_argument('--output', type=str, default='ablation',
                       help='Output directory for results')
    
    # Parallel execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (faster execution)')
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output and show progress bar only')
    
    # Output and reporting
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate markdown report after experiment')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualizations')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration from file
        logger.info(f"Loading configuration from {args.config}")
        config = SimpleClusterConfig.from_file(args.config)
        base_config = config.to_dict()

        # Apply command-line overrides
        if args.train_data:
            logger.info(f"Overriding train_data_path: {args.train_data}")
            base_config['train_data_path'] = args.train_data
        if args.test_data:
            logger.info(f"Overriding test_data_path: {args.test_data}")
            base_config['test_data_path'] = args.test_data

        # Parse weight configuration
        performance_weight_range = parse_weight_range(args.weight_range)

        logger.info(f"Starting weight ablation study")
        logger.info(f"Train data: {base_config.get('train_data_path')}")
        logger.info(f"Test data: {base_config.get('test_data_path')}")
        logger.info(f"Performance weight range: {performance_weight_range}")
        logger.info(f"Step size: {args.step_size}")

        # Initialize embedding cache
        global EMBEDDING_CACHE
        EMBEDDING_CACHE.clear()
        logger.info("Initialized embedding cache (dictionary-based)")

        # Pre-fill embedding cache with all queries
        prefill_embedding_cache(
            train_data_path=base_config.get('train_data_path'),
            test_data_path=base_config.get('test_data_path'),
            config=config,
            logger=logger
        )

        # Initialize and run weight ablation
        weight_ablation = WeightAblation(output_dir=args.output)
        results = weight_ablation.run_weight_ablation(
            base_config=base_config,
            performance_weight_range=performance_weight_range,
            step_size=args.step_size,
            load_baseline=True,
            parallel=args.parallel,
            max_workers=args.workers,
            quiet=args.quiet
        )
        
        # Generate report if requested
        report_path = None
        if args.generate_report:
            logger.info("Generating markdown report")
            report_path = generate_weight_report(results, args.output)
            print(f"\n📄 Report generated: {report_path}")
        
        # Print summary
        successful_experiments = results.get('successful_experiments', 0)
        total_experiments = results.get('total_experiments', 0)

        print(f"\n✅ Weight ablation study completed successfully!")
        print(f"📊 Experiments: {successful_experiments}/{total_experiments} successful")
        print(f"📁 Results saved to: {args.output}/results/")
        print(f"💾 Embedding cache size: {len(EMBEDDING_CACHE)} entries")
        
        if not args.no_visualizations and results.get('figure_paths'):
            print(f"📈 Visualizations saved to: {args.output}/figures/")
        
        # Show key findings
        analysis = results.get('analysis', {})
        optimal_configs = analysis.get('optimal_configurations', {})
        
        if optimal_configs.get('best_accuracy'):
            best_config = optimal_configs['best_accuracy']
            perf_weight = best_config.get('performance_weight', 0)
            cost_sens = best_config.get('cost_sensitivity', 0)
            print(f"\n🎯 Key Findings:")
            print(f"   Best accuracy config: perf_weight={perf_weight:.2f}, cost_sens={cost_sens:.2f} (accuracy: {best_config.get('accuracy', 0):.1%})")
        
        if optimal_configs.get('best_cost_efficiency'):
            eff_config = optimal_configs['best_cost_efficiency']
            perf_weight = eff_config.get('performance_weight', 0)
            cost_sens = eff_config.get('cost_sensitivity', 0)
            print(f"   Most efficient config: perf_weight={perf_weight:.2f}, cost_sens={cost_sens:.2f} (efficiency: {eff_config.get('cost_efficiency', 0):.2f})")
        
        # Pareto frontier info
        pareto_analysis = analysis.get('pareto_analysis', {})
        if pareto_analysis.get('pareto_frontier_points'):
            pareto_count = pareto_analysis['pareto_frontier_points']
            print(f"   Pareto optimal configs: {pareto_count} out of {successful_experiments}")
        
    except Exception as e:
        logger.error(f"Weight ablation study failed: {e}")
        sys.exit(1)


def generate_weight_report(results, output_dir):
    """Generate a focused weight ablation report."""
    report_lines = []
    
    timestamp = results.get('timestamp', datetime.now().isoformat())
    weight_configs = results.get('weight_configurations', [])
    analysis = results.get('analysis', {})
    
    # Header
    report_lines.extend([
        f"# Cost/Performance Weight Ablation Study Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Experiment:** Balance Cluster Router weight balance optimization",
        f"",
        f"---",
        f""
    ])
    
    # Experiment overview
    report_lines.extend([
        f"## 📋 Experiment Overview",
        f"",
        f"**Objective:** Find optimal balance between performance and cost optimization",
        f"**Parameters Varied:** performance_weight, cost_sensitivity (constrained: sum = 1.0)",
        f"**Configurations Tested:** {len(weight_configs)} weight combinations",
        f"**Success Rate:** {results.get('successful_experiments', 0)}/{results.get('total_experiments', 0)}",
        f"",
        f"---",
        f""
    ])
    
    # Pareto frontier analysis
    pareto_analysis = analysis.get('pareto_analysis', {})
    if pareto_analysis:
        pareto_count = pareto_analysis.get('pareto_frontier_points', 0)
        pareto_configs = pareto_analysis.get('pareto_configurations', [])
        
        report_lines.extend([
            f"## 🎯 Pareto Frontier Analysis",
            f"",
            f"**Pareto Optimal Configurations:** {pareto_count} out of {len(weight_configs)}",
            f"",
            f"These configurations represent the best trade-offs between accuracy and cost:",
            f""
        ])
        
        for i, config in enumerate(pareto_configs[:5]):  # Show top 5
            perf_w = config.get('performance_weight', 0)
            cost_s = config.get('cost_sensitivity', 0) 
            acc = config.get('accuracy', 0)
            cost = config.get('avg_cost', 0)
            report_lines.append(f"- **Config {i+1}:** perf_weight={perf_w:.2f}, cost_sens={cost_s:.2f} → accuracy={acc:.1%}, cost=${cost:.4f}")
        
        report_lines.extend([f"", f"---", f""])
    
    # Save report
    report_filename = f"weight_ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = Path(output_dir) / "reports" / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)


if __name__ == "__main__":
    main()
