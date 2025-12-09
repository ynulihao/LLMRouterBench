import argparse
import sys
from pathlib import Path
from loguru import logger

from .config_loader import ConfigLoader
from .planner import RunPlanner
from .runner import BenchmarkRunner
from .storage import ResultsStorage


def setup_logging(level: str):
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper()
    )


def cmd_run(args):
    """Execute benchmark runs"""
    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Setup logging
    setup_logging(args.log_level or config.run.log_level)

    # Override config with command-line arguments
    if args.overwrite:
        logger.info("Command-line --overwrite flag enabled, overriding config setting")
        config.run.overwrite = True

    # Initialize storage
    storage = ResultsStorage(config.run.output_dir)
    
    # Create planner and generate run plan
    planner = RunPlanner(config, storage)
    
    # Validate data availability
    if not planner.validate_data_availability():
        logger.error("Data validation failed")
        return 1

    # Generate run plan
    plans = planner.generate_run_plan()
    planner.print_plan_summary(plans)
    
    if not plans:
        logger.info("No runs needed")
        return 0

    # Confirm execution (if not in non-interactive mode)
    if not args.yes:
        response = input(f"Execute {len(plans)} runs? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Execution cancelled")
            return 0
    
    # Execute runs
    runner = BenchmarkRunner(config, storage)
    results = runner.run_all(plans)
    
    # Print summary
    logger.info("=== Execution Summary ===")
    logger.info(f"Total runs: {results['total_runs']}")
    logger.info(f"Successful: {results['successful_runs']}")
    logger.info(f"Failed: {results['failed_runs']}")
    
    if results['failed_runs'] > 0:
        logger.warning("Some runs failed. Check logs for details.")
        return 1
    
    return 0


def cmd_list(args):
    """List existing benchmark results"""
    storage = ResultsStorage(args.output_dir)
    results = storage.list_results()

    if not results:
        print("No results found")
        return 0

    # Sort based on order parameter
    if args.order == 'dataset':
        results = sorted(results, key=lambda x: (x['dataset_id'], x['model_name']))
    elif args.order == 'model':
        results = sorted(results, key=lambda x: (x['model_name'], x['dataset_id']))
    elif args.order == 'updated':
        # Keep the default sort from storage (by updated_at, descending)
        pass

    # Print results table
    print(f"{'Benchmark':<20} {'Split':<15} {'Model':<36} {'Count':<8} {'Updated':<30}")
    print("-" * 128)

    for result in results:
        print(f"{result['dataset_id']:<20} {result['split']:<15} {result['model_name']:<36} {result['counts']:<8} {result['updated_at']:<30}")

    total_counts = sum(result['counts'] for result in results)
    print(f"\nTotal: {len(results)} results, {total_counts} records")
    return 0


def cmd_clean(args):
    """Clean benchmark results"""
    storage = ResultsStorage(args.output_dir)
    
    if args.all:
        # Clean all results
        results_dir = Path(args.output_dir) / "bench"
        if results_dir.exists():
            import shutil
            shutil.rmtree(results_dir)
            print("All results cleaned")
        else:
            print("No results directory found")
    else:
        # Clean specific dataset/model
        if not args.dataset:
            print("Please specify --dataset or use --all")
            return 1
        
        dataset_dir = Path(args.output_dir) / "bench" / args.dataset
        if dataset_dir.exists():
            import shutil
            shutil.rmtree(dataset_dir)
            print(f"Results for {args.dataset} cleaned")
        else:
            print(f"No results found for {args.dataset}")
    
    return 0


def cmd_info(args):
    """Show configuration and system information"""
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
        
        print("=== Configuration Info ===")
        print(f"Models: {len(config.models)}")
        for model in config.models:
            print(f"  - {model.name} ({model.api_model_name})")
        
        print(f"\nDatasets: {len(config.datasets)}")
        for dataset in config.datasets:
            print(f"  - {dataset.dataset_id}")
        
        print(f"\nRun Config:")
        print(f"  - Concurrency: {config.run.concurrency}")
        print(f"  - Output Dir: {config.run.output_dir}")
        print(f"  - Overwrite: {config.run.overwrite}")
        
        # Check data availability
        storage = ResultsStorage(config.run.output_dir)
        planner = RunPlanner(config, storage)
        
        print("\n=== Data Availability ===")
        for dataset_config in config.datasets:
            try:
                all_splits = ConfigLoader.get_evaluator_splits(dataset_config.dataset_id)
                target_splits = ConfigLoader.filter_splits(
                    all_splits=all_splits,
                    target_splits=dataset_config.splits
                )
                print(f"  - {dataset_config.dataset_id}: {len(target_splits)} splits available")
            except Exception as e:
                print(f"  - {dataset_config.dataset_id}: ERROR - {e}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Data Collector - Benchmark execution system for LLM evaluation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute benchmark runs')
    run_parser.add_argument('config', help='Path to configuration YAML file')
    run_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt (non-interactive mode)')
    run_parser.add_argument('--overwrite', action='store_true', help='Force overwrite existing results (override config setting)')
    run_parser.add_argument('--log-level', help='Override log level (DEBUG, INFO, WARNING, ERROR)')
    run_parser.set_defaults(func=cmd_run)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List existing results')
    list_parser.add_argument('--output-dir', default='results', help='Results directory')
    list_parser.add_argument('--order', default='dataset', choices=['dataset', 'model', 'updated'], help='Sort order (default: dataset)')
    list_parser.set_defaults(func=cmd_list)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean results')
    clean_parser.add_argument('--output-dir', default='results', help='Results directory')
    clean_parser.add_argument('--dataset', help='Clean specific dataset')
    clean_parser.add_argument('--all', action='store_true', help='Clean all results')
    clean_parser.set_defaults(func=cmd_clean)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show configuration info')
    info_parser.add_argument('config', help='Path to configuration YAML file')
    info_parser.set_defaults(func=cmd_info)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())