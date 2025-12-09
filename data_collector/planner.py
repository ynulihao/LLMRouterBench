from typing import List, Tuple, NamedTuple
from loguru import logger

from .config_loader import BenchmarkConfig, ConfigLoader
from .storage import ResultsStorage


class RunPlan(NamedTuple):
    """Represents a single run unit: (dataset_id, split, model_name)"""
    dataset_id: str
    split: str
    model_name: str
    run_key: str  # Formatted as "dataset_id/split/model_name"


class RunPlanner:
    """Generate execution plans for benchmark runs"""
    
    def __init__(self, config: BenchmarkConfig, storage: ResultsStorage):
        self.config = config
        self.storage = storage
    
    def generate_run_plan(self) -> List[RunPlan]:
        """Generate complete run plan with deduplication"""
        plans = []
        
        for dataset_config in self.config.datasets:
            # Get available splits from evaluator
            try:
                all_splits = ConfigLoader.get_evaluator_splits(dataset_config.dataset_id)
            except Exception as e:
                logger.warning(f"Skipping dataset {dataset_config.dataset_id}: {e}")
                continue
            
            # Filter splits based on configuration
            target_splits = ConfigLoader.filter_splits(
                all_splits=all_splits,
                target_splits=dataset_config.splits
            )
            
            if not target_splits:
                logger.warning(f"No splits found for dataset {dataset_config.dataset_id}")
                continue
            
            logger.info(f"Dataset {dataset_config.dataset_id}: {len(target_splits)} splits found")

            # Generate plans for each split × model combination
            for split in target_splits:
                # Load data and calculate fingerprint for this split (used for all models)
                try:
                    from evaluation.factory import EvaluatorFactory
                    factory = EvaluatorFactory()
                    evaluator = factory.get_evaluator(dataset_config.dataset_id)
                    data = evaluator.load_data(split)
                    current_fingerprint = self.storage.calculate_data_fingerprint(data)
                except Exception as e:
                    logger.warning(f"Failed to load data for fingerprint check ({dataset_config.dataset_id}/{split}): {e}")
                    current_fingerprint = ""

                for model_config in self.config.models:
                    run_key = f"{dataset_config.dataset_id}/{split}/{model_config.name}"

                    # Check if run is needed (deduplication + fingerprint check)
                    if self.storage.needs_run(
                        dataset_config.dataset_id,
                        split,
                        model_config.name,
                        current_fingerprint=current_fingerprint,
                        overwrite=self.config.run.overwrite
                    ):
                        plans.append(RunPlan(
                            dataset_id=dataset_config.dataset_id,
                            split=split,
                            model_name=model_config.name,
                            run_key=run_key
                        ))
                    else:
                        logger.info(f"Skipping existing result: {run_key}")
        
        logger.info(f"Generated {len(plans)} run plans")
        return plans
    
    def validate_data_availability(self) -> bool:
        """Validate that all required datasets and splits are available"""
        missing_data = []
        
        for dataset_config in self.config.datasets:
            try:
                all_splits = ConfigLoader.get_evaluator_splits(dataset_config.dataset_id)
                
                # Check if any target splits are available
                target_splits = ConfigLoader.filter_splits(
                    all_splits=all_splits,
                    target_splits=dataset_config.splits
                )
                
                if not target_splits:
                    missing_data.append(f"No valid splits found for {dataset_config.dataset_id}")
                
            except Exception:
                missing_data.append(f"Evaluator not found for dataset: {dataset_config.dataset_id}")
        
        if missing_data:
            logger.error("Data validation failed:")
            for error in missing_data:
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def print_plan_summary(self, plans: List[RunPlan]):
        """Print a summary of the execution plan"""
        if not plans:
            logger.warning("No runs planned")
            return
        
        # Group by dataset
        by_dataset = {}
        for plan in plans:
            if plan.dataset_id not in by_dataset:
                by_dataset[plan.dataset_id] = []
            by_dataset[plan.dataset_id].append(plan)
        
        logger.info("=== Run Plan Summary ===")
        logger.info(f"Total runs: {len(plans)}")
        logger.info(f"Datasets: {len(by_dataset)}")
        logger.info(f"Models: {len(self.config.models)}")
        logger.info(f"Concurrency: {self.config.run.concurrency}")
        
        for dataset_id, dataset_plans in by_dataset.items():
            splits = set(plan.split for plan in dataset_plans)
            models = set(plan.model_name for plan in dataset_plans)
            logger.info(f"  {dataset_id}: {len(splits)} splits × {len(models)} models = {len(dataset_plans)} runs")