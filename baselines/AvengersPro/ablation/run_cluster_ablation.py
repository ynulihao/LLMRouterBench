#!/usr/bin/env python3
"""
Cluster Ablation Runner

Independent script for running n_clusters ablation study only.
Provides focused interface and configuration for cluster parameter optimization.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cluster_ablation import ClusterAblation
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


def parse_cluster_range(cluster_range_str):
    """Parse cluster range string into tuple."""
    if not cluster_range_str:
        return None
    
    parts = cluster_range_str.split(',')
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError("Cluster range must be 'min,max,step' format")


def main():
    """Main entry point for cluster ablation."""
    parser = argparse.ArgumentParser(
        description='N-Clusters Ablation Study for Balance Cluster Router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cluster ablation with default range (8-80)
  python run_cluster_ablation.py --config config/ablation_cluster_config.json

  # Custom cluster range
  python run_cluster_ablation.py --config config/ablation_cluster_config.json --cluster-range 16,64,8

  # Parallel execution with progress bar (recommended)
  python run_cluster_ablation.py --config config/ablation_cluster_config.json --parallel --quiet

  # Custom number of workers
  python run_cluster_ablation.py --config config/ablation_cluster_config.json --parallel --workers 8

  # Override train/test data paths
  python run_cluster_ablation.py --config config/ablation_cluster_config.json --train-data data/custom_train.jsonl --test-data data/custom_test.jsonl
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

    # Cluster ablation specific arguments
    parser.add_argument('--cluster-range', type=str,
                       help='Cluster range as "min,max,step" (e.g., "8,80,8")')

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

        # Parse cluster range
        cluster_range = None
        if args.cluster_range:
            cluster_range = parse_cluster_range(args.cluster_range)

        logger.info(f"Starting cluster ablation study")
        logger.info(f"Train data: {base_config.get('train_data_path')}")
        logger.info(f"Test data: {base_config.get('test_data_path')}")
        logger.info(f"Cluster range: {cluster_range if cluster_range else 'default (8-80)'}")

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

        # Initialize and run cluster ablation
        cluster_ablation = ClusterAblation(output_dir=args.output)
        results = cluster_ablation.run_cluster_ablation(
            base_config=base_config,
            cluster_range=cluster_range,
            load_baseline=True,
            parallel=args.parallel,
            max_workers=args.workers,
            quiet=args.quiet
        )
        
        # Generate report if requested
        report_path = None
        if args.generate_report:
            logger.info("Generating markdown report")
            report_path = generate_cluster_report(results, args.output)
            print(f"\n📄 Report generated: {report_path}")
        
        # Print summary
        successful_experiments = results.get('successful_experiments', 0)
        total_experiments = results.get('total_experiments', 0)

        print(f"\n✅ Cluster ablation study completed successfully!")
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
            print(f"\n🎯 Key Findings:")
            print(f"   Best accuracy: n_clusters={best_config.get('n_clusters')} (accuracy: {best_config.get('accuracy', 0):.1%})")
        
        if optimal_configs.get('best_cost_efficiency'):
            eff_config = optimal_configs['best_cost_efficiency']
            print(f"   Most efficient: n_clusters={eff_config.get('n_clusters')} (efficiency: {eff_config.get('cost_efficiency', 0):.2f})")
        
        if optimal_configs.get('best_balanced'):
            balanced_config = optimal_configs['best_balanced']
            print(f"   Best balanced: n_clusters={balanced_config.get('n_clusters')} (score: {balanced_config.get('balanced_score', 0):.2f})")
        
        # Performance trends
        perf_analysis = analysis.get('performance_analysis', {})
        if perf_analysis.get('accuracy_trend'):
            trend = perf_analysis['accuracy_trend']
            print(f"   Performance trend: {trend}")
        
        # Baseline comparison
        baseline_comp = analysis.get('baseline_comparison', {})
        if baseline_comp.get('improvement_over_baseline'):
            improvement = baseline_comp['improvement_over_baseline']
            print(f"   Improvement over baseline: {improvement:.1%}")
        
    except Exception as e:
        logger.error(f"Cluster ablation study failed: {e}")
        sys.exit(1)


def generate_cluster_report(results, output_dir):
    """Generate a focused cluster ablation report."""
    report_lines = []
    
    timestamp = results.get('timestamp', datetime.now().isoformat())
    cluster_range = results.get('cluster_range', [])
    analysis = results.get('analysis', {})
    
    # Header
    report_lines.extend([
        f"# N-Clusters Ablation Study Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Experiment:** Balance Cluster Router n_clusters parameter optimization",
        f"",
        f"---",
        f""
    ])
    
    # Experiment overview
    report_lines.extend([
        f"## 📋 Experiment Overview",
        f"",
        f"**Objective:** Determine optimal number of clusters for query routing",
        f"**Parameter Varied:** n_clusters",
        f"**Range Tested:** {min(cluster_range)} to {max(cluster_range)} ({len(cluster_range)} configurations)",
        f"**Success Rate:** {results.get('successful_experiments', 0)}/{results.get('total_experiments', 0)}",
        f"",
        f"---",
        f""
    ])
    
    # Key findings
    optimal_configs = analysis.get('optimal_configurations', {})
    if optimal_configs:
        report_lines.extend([
            f"## 🎯 Key Findings",
            f""
        ])
        
        if optimal_configs.get('best_accuracy'):
            best_config = optimal_configs['best_accuracy']
            report_lines.extend([
                f"### Optimal Configuration for Accuracy",
                f"- **Best n_clusters:** {best_config.get('n_clusters')}",
                f"- **Achieved accuracy:** {best_config.get('accuracy', 0):.1%}",
                f"- **Cost efficiency:** {best_config.get('cost_analysis', {}).get('cost_efficiency', 0):.2f}",
                f""
            ])
        
        if optimal_configs.get('best_cost_efficiency'):
            eff_config = optimal_configs['best_cost_efficiency']
            report_lines.extend([
                f"### Optimal Configuration for Cost Efficiency",
                f"- **Best n_clusters:** {eff_config.get('n_clusters')}",
                f"- **Cost efficiency:** {eff_config.get('cost_efficiency', 0):.2f}",
                f"- **Accuracy:** {eff_config.get('accuracy', 0):.1%}",
                f""
            ])
        
        report_lines.extend([f"---", f""])
    
    # Performance analysis
    perf_analysis = analysis.get('performance_analysis', {})
    if perf_analysis:
        report_lines.extend([
            f"## 📈 Performance Analysis",
            f"",
            f"- **Accuracy range:** {perf_analysis.get('accuracy_range', 0):.3f}",
            f"- **Standard deviation:** {perf_analysis.get('accuracy_std', 0):.3f}",
            f"- **Performance trend:** {perf_analysis.get('accuracy_trend', 'unknown').title()}",
            f"",
            f"---",
            f""
        ])
    
    # Recommendations
    report_lines.extend([
        f"## 💡 Recommendations",
        f""
    ])
    
    if optimal_configs.get('best_accuracy'):
        best_n = optimal_configs['best_accuracy'].get('n_clusters')
        report_lines.append(f"- **For production deployment:** Use n_clusters = {best_n}")
    
    if perf_analysis.get('accuracy_trend') == 'increasing':
        max_tested = max(cluster_range)
        report_lines.append(f"- **Consider testing higher values:** Performance trend is increasing, try values > {max_tested}")
    elif perf_analysis.get('accuracy_trend') == 'decreasing':
        min_tested = min(cluster_range)
        report_lines.append(f"- **Consider testing lower values:** Performance trend is decreasing, try values < {min_tested}")
    
    report_lines.extend([
        f"- **Validate on different datasets** before final deployment",
        f"- **Monitor cost implications** when changing cluster count",
        f"",
        f"---",
        f"",
        f"*Report generated by Balance Cluster Router Ablation System*"
    ])
    
    # Save report
    report_filename = f"cluster_ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = Path(output_dir) / "reports" / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)


if __name__ == "__main__":
    main()
