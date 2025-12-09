"""
Statistical aggregation utilities for baseline data.

This module provides tools for computing various statistics and comparisons
across datasets, models, and benchmark results.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import asdict

from .schema import AggregatedStats, BaselineRecord
from loguru import logger


class BaselineAggregator:
    """
    Compute aggregated statistics and comparisons from baseline records.

    Provides various aggregation views:
    - By dataset: Compare models on the same dataset
    - By model: Compare datasets for the same model
    - Global: Overall statistics across all data
    """

    def __init__(self, records: List[BaselineRecord], data_loader: Optional['BaselineDataLoader'] = None):
        """
        Initialize aggregator with a list of baseline records.

        Args:
            records: List of BaselineRecord objects to aggregate
            data_loader: Optional BaselineDataLoader instance for test_mode splitting
                        and reference_models configuration
        """
        self.records = records
        self.data_loader = data_loader

        # Extract reference models from data_loader configuration
        self.reference_models = []
        if data_loader and hasattr(data_loader, 'reference_models'):
            self.reference_models = data_loader.reference_models or []

        logger.info(f"Initialized aggregator with {len(records)} records")
        if self.reference_models:
            logger.info(f"Reference models (excluded from aggregates): {self.reference_models}")

    @staticmethod
    def _compute_stats_for_group(records: List[BaselineRecord],
                                   dataset_id: str,
                                   split: str,
                                   model_name: str) -> AggregatedStats:
        """
        Compute statistics for a group of records.

        Args:
            records: Records to aggregate
            dataset_id: Dataset identifier
            split: Dataset split
            model_name: Model name

        Returns:
            AggregatedStats object
        """
        total_records = len(records)
        if total_records == 0:
            return AggregatedStats(
                dataset_id=dataset_id,
                split=split,
                model_name=model_name,
                avg_score=0.0,
                total_records=0,
                correct_records=0,
                total_cost=0.0,
                avg_cost_per_record=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                avg_prompt_tokens=0.0,
                avg_completion_tokens=0.0
            )

        correct_records = sum(1 for r in records if r.score > 0)
        avg_score = sum(r.score for r in records) / total_records

        total_cost = sum(r.cost for r in records)
        total_prompt_tokens = sum(r.prompt_tokens for r in records)
        total_completion_tokens = sum(r.completion_tokens for r in records)

        return AggregatedStats(
            dataset_id=dataset_id,
            split=split,
            model_name=model_name,
            avg_score=avg_score,
            total_records=total_records,
            correct_records=correct_records,
            total_cost=total_cost,
            avg_cost_per_record=total_cost / total_records,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            avg_prompt_tokens=total_prompt_tokens / total_records,
            avg_completion_tokens=total_completion_tokens / total_records
        )

    def _compute_oracle_stats(self, exclude_models: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute oracle accuracy for each dataset/split.

        Oracle accuracy: For each question, if ANY model got it correct (score > 0),
        count it as correct. This represents the upper bound of ensemble performance.

        Args:
            exclude_models: List of model names to exclude from Oracle calculation (e.g., reference models)

        Returns:
            Dict mapping dataset_key (dataset_id/split) to oracle accuracy
        """
        # Set of models to exclude
        exclude_set = set(exclude_models) if exclude_models else set()

        # Group by dataset/split and record_index, excluding specified models
        questions = defaultdict(lambda: defaultdict(list))

        for record in self.records:
            if record.model_name not in exclude_set:
                dataset_key = f"{record.dataset_id}/{record.split}"
                questions[dataset_key][record.record_index].append(record.score)

        # Calculate oracle accuracy for each dataset
        oracle_accuracies = {}
        for dataset_key, records_by_index in questions.items():
            total_questions = len(records_by_index)
            oracle_correct = sum(1 for scores in records_by_index.values() if any(s > 0 for s in scores))
            oracle_accuracies[dataset_key] = oracle_correct / total_questions if total_questions > 0 else 0.0

        return oracle_accuracies

    def _compute_oracle_cost_by_dataset(self, cost_metric: str = 'total_cost', exclude_models: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute oracle cost for each dataset/split.

        Oracle cost: For each question, select the lowest cost among:
        - Correct answers (score > 0) if any exist
        - All answers if no correct answer exists

        This represents the minimum cost achievable if we could perfectly route
        each question to the cheapest correct model (or cheapest model if none correct).

        Args:
            cost_metric: Cost metric to use ('total_cost' or 'avg_cost_per_record')
            exclude_models: List of model names to exclude from calculation (e.g., reference models)

        Returns:
            Dict mapping dataset_key (dataset_id/split) to oracle cost
        """
        # Set of models to exclude
        exclude_set = set(exclude_models) if exclude_models else set()

        # Group by dataset/split and record_index, excluding specified models
        questions = defaultdict(lambda: defaultdict(list))

        for record in self.records:
            if record.model_name not in exclude_set:
                dataset_key = f"{record.dataset_id}/{record.split}"
                questions[dataset_key][record.record_index].append({
                    'score': record.score,
                    'cost': record.cost
                })

        # Calculate oracle cost for each dataset
        oracle_costs = {}
        for dataset_key, records_by_index in questions.items():
            total_oracle_cost = 0.0
            total_questions = len(records_by_index)

            for record_index, records_list in records_by_index.items():
                # Separate correct and incorrect answers
                correct_records = [r for r in records_list if r['score'] > 0]

                if correct_records:
                    # If any correct answer exists, choose the cheapest correct one
                    min_cost = min(r['cost'] for r in correct_records)
                else:
                    # If no correct answer, choose the cheapest overall
                    min_cost = min(r['cost'] for r in records_list)

                total_oracle_cost += min_cost

            # Return based on cost_metric
            if cost_metric == 'total_cost':
                oracle_costs[dataset_key] = total_oracle_cost
            elif cost_metric == 'avg_cost_per_record':
                oracle_costs[dataset_key] = total_oracle_cost / total_questions if total_questions > 0 else 0.0
            else:
                raise ValueError(f"Invalid cost_metric: {cost_metric}")

        return oracle_costs

    def _compute_max_expert_cost_by_dataset(self, cost_metric: str = 'total_cost', exclude_models: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute cost of the best performing model (Max Expert) for each dataset/split.

        Max Expert cost: For each dataset, find the model with highest avg_score,
        and return its cost.

        Args:
            cost_metric: Cost metric to use ('total_cost' or 'avg_cost_per_record')
            exclude_models: List of model names to exclude from calculation (e.g., reference models)

        Returns:
            Dict mapping dataset_key (dataset_id/split) to max expert cost
        """
        # Set of models to exclude
        exclude_set = set(exclude_models) if exclude_models else set()

        # Get aggregated stats by dataset and model
        by_dataset_model = self.aggregate_by_dataset_and_model()

        max_expert_costs = {}
        for dataset_key, models in by_dataset_model.items():
            # Filter out excluded models
            eligible_models = {
                model_name: stats
                for model_name, stats in models.items()
                if model_name not in exclude_set
            }

            if not eligible_models:
                continue

            # Find the model with highest avg_score
            best_model_name = max(
                eligible_models.keys(),
                key=lambda m: eligible_models[m].avg_score
            )
            best_model_stats = eligible_models[best_model_name]

            # Get the corresponding cost
            max_expert_costs[dataset_key] = getattr(best_model_stats, cost_metric)

        return max_expert_costs

    def _sample_random_router_once(self, random_seed: int = 42, exclude_models: Optional[List[str]] = None) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
        """
        Sample random router once for all samples (global sampling).

        For each sample, randomly select one model with uniform probability.
        This ensures consistency across all computations.

        Args:
            random_seed: Random seed for reproducible sampling
            exclude_models: List of model names to exclude from sampling (e.g., reference models)

        Returns:
            Dict mapping (dataset_id, split, record_index) to sampled record info:
            {
                'score': float,
                'cost': float,
                'prompt_tokens': int,
                'completion_tokens': int
            }
        """
        import random

        # Create independent random number generator for reproducibility
        rng = random.Random(random_seed)

        # Set of models to exclude
        exclude_set = set(exclude_models) if exclude_models else set()

        # Group all records by sample, excluding specified models
        questions = defaultdict(list)
        for record in self.records:
            if record.model_name not in exclude_set:
                key = (record.dataset_id, record.split, record.record_index)
                questions[key].append(record)

        # Sample one record for each sample
        sampled_data = {}
        for key, records in questions.items():
            if records:  # Only sample if there are available records
                sampled_record = rng.choice(records)
                sampled_data[key] = {
                    'score': sampled_record.score,
                    'cost': sampled_record.cost,
                    'prompt_tokens': sampled_record.prompt_tokens,
                    'completion_tokens': sampled_record.completion_tokens
                }

        return sampled_data

    def _compute_random_router_by_dataset(self, sampled_data: Dict[Tuple[str, str, int], Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute random router accuracy for each dataset/split using pre-sampled data.

        Args:
            sampled_data: Pre-sampled data from _sample_random_router_once()
                         Dict mapping (dataset_id, split, record_index) to sampled record info

        Returns:
            Dict mapping dataset_key (dataset_id/split) to random router accuracy
        """
        # Group sampled scores by dataset
        dataset_scores = defaultdict(list)
        for (dataset_id, split, record_index), data in sampled_data.items():
            dataset_key = f"{dataset_id}/{split}"
            dataset_scores[dataset_key].append(data['score'])

        # Calculate average for each dataset
        random_router_accuracies = {}
        for dataset_key, scores in dataset_scores.items():
            random_router_accuracies[dataset_key] = sum(scores) / len(scores) if scores else 0.0

        return random_router_accuracies

    def _compute_random_router_cost_by_dataset(self, sampled_data: Dict[Tuple[str, str, int], Dict[str, Any]], cost_metric: str = 'total_cost') -> Dict[str, float]:
        """
        Compute random router cost for each dataset/split using pre-sampled data.

        Args:
            sampled_data: Pre-sampled data from _sample_random_router_once()
                         Dict mapping (dataset_id, split, record_index) to sampled record info
            cost_metric: Cost metric to use ('total_cost' or 'avg_cost_per_record')

        Returns:
            Dict mapping dataset_key (dataset_id/split) to random router cost
        """
        # Group sampled costs by dataset
        dataset_costs = defaultdict(list)
        for (dataset_id, split, record_index), data in sampled_data.items():
            dataset_key = f"{dataset_id}/{split}"
            dataset_costs[dataset_key].append(data['cost'])

        # Calculate cost for each dataset
        random_router_costs = {}
        for dataset_key, costs in dataset_costs.items():
            if cost_metric == 'total_cost':
                random_router_costs[dataset_key] = sum(costs)
            elif cost_metric == 'avg_cost_per_record':
                random_router_costs[dataset_key] = sum(costs) / len(costs) if costs else 0.0
            else:
                raise ValueError(f"Invalid cost_metric: {cost_metric}")

        return random_router_costs

    def _compute_sample_level_avg(self, sampled_data: Optional[Dict[Tuple[str, str, int], Dict[str, Any]]] = None, exclude_models: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute sample-level average accuracy for all models and aggregate rows.

        Sample-level average: sum(all scores) / total samples across all datasets.
        This differs from dataset-level average which is: mean of dataset averages.

        Args:
            sampled_data: Pre-sampled data for Random Router (optional)
                         Dict mapping (dataset_id, split, record_index) to sampled record info
            exclude_models: List of model names to exclude from aggregate calculations
                            (e.g., reference models)

        Returns:
            Dict mapping row name to sample-level average:
            - Model names: their sample-level accuracy
            - 'AVG': average of MAIN models' sample-level avg (excluding exclude_models)
            - 'Random Router': accuracy when randomly sampling one model for each sample
            - 'Max Expert': sample-level accuracy of per-dataset best MAIN models
            - 'Oracle': oracle sample-level accuracy (excluding specified models)
        """
        # Delegate to filtered implementation with no dataset restriction
        return self._compute_sample_level_avg_by_filter(
            dataset_filter=None,
            exclude_models=exclude_models,
            sampled_data=sampled_data
        )

    def _separate_datasets(self, dataset_keys: List[str], ood_datasets: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Separate dataset keys into in-distribution and OOD groups.

        Args:
            dataset_keys: List of dataset keys (e.g., ["aime/hybrid", "brainteaser/test"])
            ood_datasets: List of OOD dataset IDs to filter

        Returns:
            Tuple of (in_distribution_keys, ood_keys)
        """
        if not ood_datasets:
            return sorted(dataset_keys), []

        ood_set = set(ood_datasets)
        in_dist_keys = []
        ood_keys = []

        for key in dataset_keys:
            dataset_id = key.split('/')[0]
            if dataset_id in ood_set:
                ood_keys.append(key)
            else:
                in_dist_keys.append(key)

        return sorted(in_dist_keys), sorted(ood_keys)

    def _compute_sample_level_avg_by_filter(
        self,
        dataset_filter: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        sampled_data: Optional[Dict[Tuple[str, str, int], Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Compute sample-level average accuracy filtered by dataset list.

        Args:
            dataset_filter: List of dataset IDs to include (None means all)
            exclude_models: List of model names to exclude
            sampled_data: Pre-sampled data for Random Router (optional)

        Returns:
            Dict mapping row name to sample-level average for filtered datasets
        """
        # Get unique model names from records
        model_names = sorted(set(r.model_name for r in self.records))

        # Filter records by dataset
        filtered_records = self.records
        if dataset_filter:
            dataset_set = set(dataset_filter)
            filtered_records = [r for r in self.records if r.dataset_id in dataset_set]

        if not filtered_records:
            # Return zeros if no records match
            result = {m: 0.0 for m in model_names}
            result['AVG'] = 0.0
            result['Max Expert'] = 0.0
            result['Random Router'] = 0.0
            result['Oracle'] = 0.0
            return result

        # Compute sample-level avg for each model
        sample_level_avg = {}
        for model_name in model_names:
            model_records = [r for r in filtered_records if r.model_name == model_name]
            if model_records:
                total_score = sum(r.score for r in model_records)
                total_samples = len(model_records)
                sample_level_avg[model_name] = total_score / total_samples
            else:
                sample_level_avg[model_name] = 0.0

        # Compute aggregate statistics over MAIN models only (exclude reference models)
        exclude_set = set(exclude_models) if exclude_models else set()
        main_model_values = [v for m, v in sample_level_avg.items() if m not in exclude_set]

        if main_model_values:
            sample_level_avg['AVG'] = sum(main_model_values) / len(main_model_values)
        else:
            sample_level_avg['AVG'] = 0.0

        # Compute Max Expert sample-level avg using per-dataset best MAIN model
        # For each dataset/split, select the main model with highest average score,
        # then average its scores across all samples.
        from collections import defaultdict as _dd

        dataset_model_scores: Dict[str, Dict[str, List[float]]] = _dd(lambda: _dd(list))  # type: ignore[var-annotated]
        for record in filtered_records:
            if record.model_name in exclude_set:
                continue
            dataset_key = f"{record.dataset_id}/{record.split}"
            dataset_model_scores[dataset_key][record.model_name].append(record.score)

        best_scores: List[float] = []
        for dataset_key, model_scores in dataset_model_scores.items():
            if not model_scores:
                continue
            # Choose model with highest average score on this dataset
            best_model, scores = max(
                model_scores.items(),
                key=lambda kv: (sum(kv[1]) / len(kv[1])) if kv[1] else float("-inf")
            )
            # Accumulate all scores for the best model on this dataset
            best_scores.extend(scores)

        if best_scores:
            sample_level_avg['Max Expert'] = sum(best_scores) / len(best_scores)
        else:
            sample_level_avg['Max Expert'] = 0.0

        # Compute Random Router sample-level avg using pre-sampled data
        if sampled_data and dataset_filter:
            dataset_set = set(dataset_filter)
            filtered_sampled = {
                key: data for key, data in sampled_data.items()
                if key[0] in dataset_set  # key[0] is dataset_id
            }
            if filtered_sampled:
                total_score = sum(data['score'] for data in filtered_sampled.values())
                sample_level_avg['Random Router'] = total_score / len(filtered_sampled)
            else:
                sample_level_avg['Random Router'] = 0.0
        elif sampled_data:
            total_score = sum(data['score'] for data in sampled_data.values())
            sample_level_avg['Random Router'] = total_score / len(sampled_data) if sampled_data else 0.0
        else:
            sample_level_avg['Random Router'] = 0.0

        # Compute Oracle sample-level avg (excluding specified models)
        questions_filtered = defaultdict(list)
        for record in filtered_records:
            if record.model_name not in exclude_set:
                key = (record.dataset_id, record.split, record.record_index)
                questions_filtered[key].append(record.score)

        oracle_correct = sum(1 for scores in questions_filtered.values() if any(s > 0 for s in scores))
        total_samples = len(questions_filtered)
        sample_level_avg['Oracle'] = oracle_correct / total_samples if total_samples > 0 else 0.0

        return sample_level_avg

    def _format_performance_table(self,
                                   perf_df: 'pd.DataFrame',
                                   score_as_percent: bool = True,
                                   precision: int = 2) -> 'pd.DataFrame':
        """
        Format performance table for display.

        Args:
            perf_df: Performance DataFrame from to_summary_table()
            score_as_percent: Display scores as percentages (e.g., 85.00 instead of 0.85)
                             Default is True (percentage format)
            precision: Number of decimal places for numerical values (default: 2)

        Returns:
            Formatted DataFrame with string values
        """
        perf_display = perf_df.copy()

        # Convert numeric values to formatted strings (default: percentage format)
        for col in perf_display.columns:
            for idx in perf_display.index:
                val = perf_display.loc[idx, col]
                if val != "-" and isinstance(val, (int, float)):
                    if score_as_percent:
                        perf_display.loc[idx, col] = f"{val * 100:.{precision}f}"
                    else:
                        perf_display.loc[idx, col] = f"{val:.{precision}f}"

        return perf_display

    def _format_cost_table(self,
                           cost_df: 'pd.DataFrame',
                           precision: int = 2) -> 'pd.DataFrame':
        """
        Format cost table for display.

        Args:
            cost_df: Cost DataFrame from to_summary_table()
            precision: Number of decimal places for numerical values

        Returns:
            Formatted DataFrame with string values
        """
        cost_display = cost_df.copy()

        # Format numeric values with specified precision
        for col in cost_display.columns:
            for idx in cost_display.index:
                val = cost_display.loc[idx, col]
                if val != "-" and isinstance(val, (int, float)):
                    cost_display.loc[idx, col] = f"{val:.{precision}f}"

        return cost_display

    def aggregate_by_dataset_and_model(self) -> Dict[str, Dict[str, AggregatedStats]]:
        """
        Aggregate statistics by dataset/split and model.

        Returns:
            Nested dict: {dataset/split: {model_name: AggregatedStats}}

        Example:
            >>> agg = BaselineAggregator(records)
            >>> stats = agg.aggregate_by_dataset_and_model()
            >>> print(stats['aime/hybrid']['gpt-4'].avg_score)
            0.85
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for record in self.records:
            key = f"{record.dataset_id}/{record.split}"
            grouped[key][record.model_name].append(record)

        results = {}
        for dataset_key, models in grouped.items():
            results[dataset_key] = {}
            dataset_id, split = dataset_key.split('/')
            for model_name, records in models.items():
                stats = self._compute_stats_for_group(
                    records, dataset_id, split, model_name
                )
                results[dataset_key][model_name] = stats

        return results

    def aggregate_by_model(self) -> Dict[str, List[AggregatedStats]]:
        """
        Aggregate statistics grouped by model (across all datasets).

        Returns:
            Dict mapping model_name to list of AggregatedStats (one per dataset)

        Example:
            >>> agg = BaselineAggregator(records)
            >>> model_stats = agg.aggregate_by_model()
            >>> for stats in model_stats['gpt-4']:
            ...     print(f"{stats.dataset_id}: {stats.avg_score}")
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for record in self.records:
            key = f"{record.dataset_id}/{record.split}"
            grouped[record.model_name][key].append(record)

        results = {}
        for model_name, datasets in grouped.items():
            results[model_name] = []
            for dataset_key, records in datasets.items():
                dataset_id, split = dataset_key.split('/')
                stats = self._compute_stats_for_group(
                    records, dataset_id, split, model_name
                )
                results[model_name].append(stats)

        return results

    def aggregate_by_dataset(self) -> Dict[str, List[AggregatedStats]]:
        """
        Aggregate statistics grouped by dataset (across all models).

        Returns:
            Dict mapping dataset/split to list of AggregatedStats (one per model)

        Example:
            >>> agg = BaselineAggregator(records)
            >>> dataset_stats = agg.aggregate_by_dataset()
            >>> for stats in dataset_stats['aime/hybrid']:
            ...     print(f"{stats.model_name}: {stats.avg_score}")
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for record in self.records:
            key = f"{record.dataset_id}/{record.split}"
            grouped[key][record.model_name].append(record)

        results = {}
        for dataset_key, models in grouped.items():
            results[dataset_key] = []
            dataset_id, split = dataset_key.split('/')
            for model_name, records in models.items():
                stats = self._compute_stats_for_group(
                    records, dataset_id, split, model_name
                )
                results[dataset_key].append(stats)

        return results

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Compute global statistics across all records.

        Returns:
            Dict with global statistics

        Example:
            >>> agg = BaselineAggregator(records)
            >>> global_stats = agg.get_global_stats()
            >>> print(f"Overall accuracy: {global_stats['avg_score']:.2%}")
        """
        if not self.records:
            return {
                'total_records': 0,
                'total_datasets': 0,
                'total_models': 0,
                'total_prompts': 0,
                'avg_score': 0.0,
                'total_cost': 0.0
            }

        total_records = len(self.records)
        avg_score = sum(r.score for r in self.records) / total_records
        total_cost = sum(r.cost for r in self.records)
        total_prompt_tokens = sum(r.prompt_tokens for r in self.records)
        total_completion_tokens = sum(r.completion_tokens for r in self.records)

        datasets = set(f"{r.dataset_id}/{r.split}" for r in self.records)
        models = set(r.model_name for r in self.records)
        unique_prompts = set((r.dataset_id, r.split, r.record_index) for r in self.records)

        return {
            'total_records': total_records,
            'total_datasets': len(datasets),
            'total_models': len(models),
            'total_prompts': len(unique_prompts),
            'avg_score': avg_score,
            'total_cost': total_cost,
            'avg_cost_per_record': total_cost / total_records,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'avg_prompt_tokens': total_prompt_tokens / total_records,
            'avg_completion_tokens': total_completion_tokens / total_records,
            'datasets': sorted(datasets),
            'models': sorted(models)
        }

    def to_summary_table(self,
                         cost_metric: str = 'total_cost',
                         test_mode: bool = False,
                         random_seed: int = 42,
                         train_ratio: float = 0.8,
                         ood_datasets: Optional[List[str]] = None) -> Tuple['pd.DataFrame', 'pd.DataFrame']:
        """
        Create two pivot tables: performance and cost across datasets and models.

        Args:
            cost_metric: Cost metric to use ('total_cost' or 'avg_cost_per_record')
            test_mode: If True, only compute statistics for test set (requires data_loader)
            random_seed: Random seed for:
                        - Random Router sampling (for each prompt, randomly select one model)
                        - Train/test splitting (used when test_mode=True)
            train_ratio: Proportion of data for training (used when test_mode=True)
            ood_datasets: List of dataset IDs to treat as OOD/test only (used when test_mode=True)

        Note:
            Reference models are loaded from data_loader.reference_models (from config file).
            These models are excluded from AVG, Random Router, Max Expert, and Oracle calculations,
            and are displayed at the bottom for comparison only.

        Returns:
            Tuple of (performance_table, cost_table) as pandas DataFrames

            Performance table rows (in order):
            - Main model names (sorted alphabetically, excluding reference_models)
            - 'AVG': Average performance across main models only
            - 'Random Router': Simulated performance when randomly selecting from main models
                              (uses actual sampling with given random_seed for reproducibility)
            - 'Max Expert': Best performance achieved by any main model for each dataset
            - 'Oracle': Best possible performance if we could pick the right main model for each question
                       (only considers main models, excluding reference_models)
            - Reference model names (sorted alphabetically, if reference_models specified)

            Cost table rows (in order):
            - Main model names (sorted alphabetically, excluding reference_models)
            - 'AVG': Average cost across main models only
            - 'Random Router': Simulated cost when randomly selecting from main models
                              (uses actual sampling with given random_seed for reproducibility)
            - 'Max Expert': Cost of the best performing main model for each dataset
            - 'Oracle': Minimum cost achievable with perfect routing (for each question,
                       select the cheapest correct model, or cheapest if none correct)
            - Reference model names (sorted alphabetically, if reference_models specified)

            Columns:
            Performance table:
            - Dataset/split names (in-distribution first, then OOD, sorted alphabetically within each group)
            - 'In-Dist-Avg': Average across in-distribution datasets (only if ood_datasets specified)
            - 'OOD-Avg': Average across OOD datasets (only if ood_datasets specified)
            - 'Dataset-Avg': Average across all datasets (each dataset weighted equally)
            - 'In-Dist-Sample-Avg': Sample-level average for in-distribution datasets (only if ood_datasets specified)
            - 'OOD-Sample-Avg': Sample-level average for OOD datasets (only if ood_datasets specified)
            - 'Sample-Avg': Average across all samples (datasets weighted by sample count)

            Cost table:
            - Dataset/split names (in-distribution first, then OOD, sorted alphabetically within each group)
            - 'In-Dist-Sum': Sum of costs across in-distribution datasets (only if ood_datasets specified)
            - 'OOD-Sum': Sum of costs across OOD datasets (only if ood_datasets specified)
            - 'Dataset-Sum': Sum of costs across all datasets

            - Performance values: avg_score (0.0 to 1.0)
            - Cost values: specified cost_metric
            - Missing values are filled with "-"

        Raises:
            ImportError: If pandas is not installed
            ValueError: If cost_metric is invalid

        Example:
            >>> agg = BaselineAggregator(records)
            >>> # Returns TWO DataFrames (performance and cost)
            >>> perf_table, cost_table = agg.to_summary_table()
            >>>
            >>> print("Performance Table:")
            >>> print(perf_table)
            >>> #                    aime/hybrid  humaneval/test  bbh/test  Dataset-Avg  Sample-Avg
            >>> # gpt-4                     0.85         0.92        0.78        0.850       0.847
            >>> # claude-3                  0.82         0.89        -           0.855       0.853
            >>> # llama-3                   0.75         0.85        0.70        0.767       0.762
            >>> # AVG                       0.807        0.887       0.740       0.824       0.821
            >>> # Random Router             0.810        0.887       0.740       0.825       0.825
            >>> # Max Expert                0.85         0.92        0.78        0.850       0.853
            >>> # Oracle                    0.90         0.95        0.82        0.890       0.887
            >>>
            >>> print("\nCost Table:")
            >>> print(cost_table)
            >>> #                    aime/hybrid  humaneval/test  bbh/test  Dataset-Sum
            >>> # gpt-4                     1.23         0.45        0.67        2.35
            >>> # claude-3                  0.98         0.38        -           1.36
            >>> # llama-3                   0.15         0.08        0.12        0.35
            >>> # AVG                       0.787        0.303       0.395       1.485
            >>> # Random Router             0.79         0.30        0.40        1.49
            >>> # Max Expert                1.23         0.45        0.67        2.35
            >>> # Oracle                    0.80         0.32        0.42        1.54
            >>>
            >>> # Use avg cost per record instead
            >>> perf_table, avg_cost_table = agg.to_summary_table(cost_metric='avg_cost_per_record')
            >>>
            >>> # With OOD datasets: columns are reordered and separate averages are computed
            >>> perf_table, cost_table = agg.to_summary_table(
            ...     ood_datasets=["brainteaser", "dailydialog"]
            ... )
            >>> # Performance table will have columns in this order:
            >>> # [in-dist datasets...] | [OOD datasets...] | In-Dist-Avg | OOD-Avg | Dataset-Avg |
            >>> # In-Dist-Sample-Avg | OOD-Sample-Avg | Sample-Avg
            >>>
            >>> # Test mode: only compute statistics for test set
            >>> perf_table, cost_table = agg.to_summary_table(
            ...     test_mode=True,
            ...     random_seed=42,
            ...     train_ratio=0.8,
            ...     ood_datasets=["brainteaser"]
            ... )
            >>>
            >>> # Reference models are loaded from config file (data_loader.reference_models)
            >>> # These models are excluded from AVG/Random Router/Max Expert/Oracle
            >>> # and displayed at the bottom for comparison only
            >>> # Config: filters.reference_models = ["gpt-4o", "claude-opus"]
            >>> perf_table, cost_table = agg.to_summary_table()
            >>> print(perf_table)
            >>> #                    aime/hybrid  humaneval/test  Dataset-Avg  Sample-Avg
            >>> # gpt-4                     0.85         0.92           0.885       0.847
            >>> # claude-3                  0.82         0.89           0.855       0.853
            >>> # llama-3                   0.75         0.85           0.800       0.762
            >>> # AVG                       0.807        0.887          0.847       0.821   <- Only main models
            >>> # Random Router             0.810        0.887          0.848       0.825   <- Only main models
            >>> # Max Expert                0.85         0.92           0.885       0.853   <- Only main models
            >>> # Oracle                    0.90         0.95           0.925       0.887   <- Only main models
            >>> # gpt-4o                    0.92         0.95           0.935       0.920   <- Reference
            >>> # claude-opus                0.90         0.93           0.915       0.905   <- Reference
        """
        # Handle test_mode: split data and only use test set
        if test_mode:
            if self.data_loader is None:
                raise ValueError(
                    "test_mode requires a BaselineDataLoader instance. "
                    "Pass data_loader when initializing BaselineAggregator:\n"
                    "  aggregator = BaselineAggregator(records, data_loader=loader)"
                )

            logger.info(f"Test mode enabled: splitting data with seed={random_seed}, ratio={train_ratio}")

            # Split data using the unified splitting logic
            _, test_records = self.data_loader.split_by_dataset_then_prompt(
                self.records,
                train_ratio=train_ratio,
                random_seed=random_seed,
                ood_datasets=ood_datasets
            )

            logger.info(f"Using test set only: {len(test_records)} records (from {len(self.records)} total)")

            # Create a temporary aggregator with test records only
            temp_aggregator = BaselineAggregator(test_records, data_loader=self.data_loader)

            # Recursively call to_summary_table with test_mode=False to avoid infinite loop
            return temp_aggregator.to_summary_table(
                cost_metric=cost_metric,
                test_mode=False,  # Important: disable test_mode to avoid recursion
                ood_datasets=ood_datasets  # Pass ood_datasets to enable OOD separation
            )

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_summary_table(). "
                "Install with: pip install pandas"
            )

        if cost_metric not in ['total_cost', 'avg_cost_per_record']:
            raise ValueError(
                f"Invalid cost_metric '{cost_metric}'. "
                f"Must be 'total_cost' or 'avg_cost_per_record'"
            )

        by_dataset_model = self.aggregate_by_dataset_and_model()

        # Build nested dicts: {model_name: {dataset_key: value}}
        performance_data = defaultdict(dict)
        cost_data = defaultdict(dict)

        for dataset_key, models in by_dataset_model.items():
            for model_name, stats in models.items():
                performance_data[model_name][dataset_key] = stats.avg_score
                cost_data[model_name][dataset_key] = getattr(stats, cost_metric)

        # Convert to DataFrames (rows=models, columns=datasets)
        performance_df = pd.DataFrame.from_dict(performance_data, orient='index')
        cost_df = pd.DataFrame.from_dict(cost_data, orient='index')

        # Separate datasets into in-distribution and OOD groups
        all_dataset_keys = list(performance_df.columns)
        in_dist_keys, ood_keys = self._separate_datasets(all_dataset_keys, ood_datasets)

        # Reorder columns: in-distribution first, then OOD
        ordered_columns = in_dist_keys + ood_keys
        performance_df = performance_df[ordered_columns]
        cost_df = cost_df[ordered_columns]

        # Sort by model name (index)
        performance_df = performance_df.sort_index()
        cost_df = cost_df.sort_index()

        # Separate main models and reference models
        reference_set = set(self.reference_models) if self.reference_models else set()
        all_model_names = [idx for idx in performance_df.index if isinstance(idx, str)]
        main_models = [m for m in all_model_names if m not in reference_set]
        ref_models = [m for m in all_model_names if m in reference_set]

        # Add AVG row (average across MAIN models only for each dataset)
        if main_models:
            performance_df.loc['AVG'] = performance_df.loc[main_models].mean(axis=0, skipna=True)
            cost_df.loc['AVG'] = cost_df.loc[main_models].mean(axis=0, skipna=True)
        else:
            performance_df.loc['AVG'] = performance_df.mean(axis=0, skipna=True)
            cost_df.loc['AVG'] = cost_df.mean(axis=0, skipna=True)

        # Sample random router once for consistency across all computations
        # Exclude reference models from sampling
        random_router_sampled_data = self._sample_random_router_once(
            random_seed=random_seed,
            exclude_models=self.reference_models
        )

        # Add Random Router row (expected performance when randomly selecting a model for each sample)
        random_router_stats = self._compute_random_router_by_dataset(random_router_sampled_data)
        random_router_row = pd.Series({col: random_router_stats.get(col, None) for col in performance_df.columns})
        performance_df.loc['Random Router'] = random_router_row

        # Add Random Router cost row (cost when randomly selecting a model for each sample)
        random_router_costs = self._compute_random_router_cost_by_dataset(random_router_sampled_data, cost_metric=cost_metric)
        random_router_cost_row = pd.Series({col: random_router_costs.get(col, None) for col in cost_df.columns})
        cost_df.loc['Random Router'] = random_router_cost_row

        # Add Max Expert cost row (cost of the best performing model for each dataset)
        # Exclude reference models from Max Expert calculation
        max_expert_costs = self._compute_max_expert_cost_by_dataset(cost_metric=cost_metric, exclude_models=self.reference_models)
        max_expert_cost_row = pd.Series({col: max_expert_costs.get(col, None) for col in cost_df.columns})
        cost_df.loc['Max Expert'] = max_expert_cost_row

        # Add Oracle cost row (minimum cost achievable with perfect routing)
        # Exclude reference models from Oracle calculation
        oracle_costs = self._compute_oracle_cost_by_dataset(cost_metric=cost_metric, exclude_models=self.reference_models)
        oracle_cost_row = pd.Series({col: oracle_costs.get(col, None) for col in cost_df.columns})
        cost_df.loc['Oracle'] = oracle_cost_row

        # Add Max Expert row (best performance for each dataset) - only for MAIN performance table
        # Exclude AVG, Random Router, and reference models when calculating max
        if main_models:
            performance_df.loc['Max Expert'] = performance_df.loc[main_models].max(axis=0, skipna=True)
        else:
            model_rows = performance_df.index[~performance_df.index.isin(['AVG', 'Random Router'])]
            performance_df.loc['Max Expert'] = performance_df.loc[model_rows].max(axis=0, skipna=True)

        # Add Oracle row (best possible performance if we could pick the right model for each question)
        # Exclude reference models from Oracle calculation
        oracle_stats = self._compute_oracle_stats(exclude_models=self.reference_models)
        oracle_row = pd.Series({col: oracle_stats.get(col, None) for col in performance_df.columns})
        performance_df.loc['Oracle'] = oracle_row

        # Compute sample-level averages for all rows
        # Exclude reference models from Oracle calculation
        sample_level_avg_dict = self._compute_sample_level_avg(
            sampled_data=random_router_sampled_data,
            exclude_models=self.reference_models
        )

        # Compute separate averages for in-distribution and OOD datasets
        if ood_keys:
            # Extract dataset IDs from keys
            in_dist_dataset_ids = [key.split('/')[0] for key in in_dist_keys]
            ood_dataset_ids = [key.split('/')[0] for key in ood_keys]

            # Compute sample-level averages for in-distribution datasets
            in_dist_sample_avg_dict = self._compute_sample_level_avg_by_filter(
                dataset_filter=in_dist_dataset_ids,
                exclude_models=self.reference_models,
                sampled_data=random_router_sampled_data
            )

            # Compute sample-level averages for OOD datasets
            ood_sample_avg_dict = self._compute_sample_level_avg_by_filter(
                dataset_filter=ood_dataset_ids,
                exclude_models=self.reference_models,
                sampled_data=random_router_sampled_data
            )

            # Add In-Dist-Avg column (average across in-distribution datasets)
            # First compute for actual models only (not aggregate rows)
            for model in main_models + ref_models:
                if model in performance_df.index:
                    performance_df.loc[model, 'In-Dist-Avg'] = performance_df.loc[model, in_dist_keys].mean(skipna=True)

            # Now compute for aggregate rows
            if main_models:
                # AVG: average of main models' In-Dist-Avg values
                performance_df.loc['AVG', 'In-Dist-Avg'] = performance_df.loc[main_models, 'In-Dist-Avg'].mean(skipna=True)
                # Max Expert: max of main models' In-Dist-Avg values
                performance_df.loc['Max Expert', 'In-Dist-Avg'] = performance_df.loc[main_models, 'In-Dist-Avg'].max(skipna=True)

            # Random Router: compute dataset-level average (not sample-level)
            if 'Random Router' in performance_df.index and in_dist_keys:
                in_dist_random_values = [random_router_stats.get(key) for key in in_dist_keys if random_router_stats.get(key) is not None]
                performance_df.loc['Random Router', 'In-Dist-Avg'] = sum(in_dist_random_values) / len(in_dist_random_values) if in_dist_random_values else None

            # Oracle: compute dataset-level average (not sample-level)
            if 'Oracle' in performance_df.index and in_dist_keys:
                in_dist_oracle_values = [oracle_stats.get(key) for key in in_dist_keys if oracle_stats.get(key) is not None]
                performance_df.loc['Oracle', 'In-Dist-Avg'] = sum(in_dist_oracle_values) / len(in_dist_oracle_values) if in_dist_oracle_values else None

            # Add OOD-Avg column (average across OOD datasets)
            # First compute for actual models only
            for model in main_models + ref_models:
                if model in performance_df.index:
                    performance_df.loc[model, 'OOD-Avg'] = performance_df.loc[model, ood_keys].mean(skipna=True)

            # Now compute for aggregate rows
            if main_models:
                # AVG: average of main models' OOD-Avg values
                performance_df.loc['AVG', 'OOD-Avg'] = performance_df.loc[main_models, 'OOD-Avg'].mean(skipna=True)
                # Max Expert: max of main models' OOD-Avg values
                performance_df.loc['Max Expert', 'OOD-Avg'] = performance_df.loc[main_models, 'OOD-Avg'].max(skipna=True)

            # Random Router: compute dataset-level average (not sample-level)
            if 'Random Router' in performance_df.index and ood_keys:
                ood_random_values = [random_router_stats.get(key) for key in ood_keys if random_router_stats.get(key) is not None]
                performance_df.loc['Random Router', 'OOD-Avg'] = sum(ood_random_values) / len(ood_random_values) if ood_random_values else None

            # Oracle: compute dataset-level average (not sample-level)
            if 'Oracle' in performance_df.index and ood_keys:
                ood_oracle_values = [oracle_stats.get(key) for key in ood_keys if oracle_stats.get(key) is not None]
                performance_df.loc['Oracle', 'OOD-Avg'] = sum(ood_oracle_values) / len(ood_oracle_values) if ood_oracle_values else None

            # Add In-Dist-Sample-Avg column using pre-computed values
            performance_df['In-Dist-Sample-Avg'] = performance_df.index.map(lambda x: in_dist_sample_avg_dict.get(x))

            # Add OOD-Sample-Avg column using pre-computed values
            performance_df['OOD-Sample-Avg'] = performance_df.index.map(lambda x: ood_sample_avg_dict.get(x))

            # Add In-Dist-Sum column for cost
            for model in main_models + ref_models:
                if model in cost_df.index:
                    cost_df.loc[model, 'In-Dist-Sum'] = cost_df.loc[model, in_dist_keys].sum(skipna=True)

            # Compute for aggregate rows
            if main_models:
                cost_df.loc['AVG', 'In-Dist-Sum'] = cost_df.loc[main_models, 'In-Dist-Sum'].mean(skipna=True)
                cost_df.loc['Max Expert', 'In-Dist-Sum'] = cost_df.loc[main_models, 'In-Dist-Sum'].max(skipna=True)

            if 'Random Router' in cost_df.index and in_dist_keys:
                in_dist_random_costs = {k: v for k, v in random_router_costs.items() if k in in_dist_keys}
                if cost_metric == 'total_cost':
                    cost_df.loc['Random Router', 'In-Dist-Sum'] = sum(in_dist_random_costs.values())
                else:
                    cost_df.loc['Random Router', 'In-Dist-Sum'] = sum(in_dist_random_costs.values()) / len(in_dist_random_costs) if in_dist_random_costs else None

            if 'Oracle' in cost_df.index and in_dist_keys:
                in_dist_oracle_costs = {k: v for k, v in oracle_costs.items() if k in in_dist_keys}
                if cost_metric == 'total_cost':
                    cost_df.loc['Oracle', 'In-Dist-Sum'] = sum(in_dist_oracle_costs.values())
                else:
                    cost_df.loc['Oracle', 'In-Dist-Sum'] = sum(in_dist_oracle_costs.values()) / len(in_dist_oracle_costs) if in_dist_oracle_costs else None

            # Add OOD-Sum column for cost
            for model in main_models + ref_models:
                if model in cost_df.index:
                    cost_df.loc[model, 'OOD-Sum'] = cost_df.loc[model, ood_keys].sum(skipna=True)

            # Compute for aggregate rows
            if main_models:
                cost_df.loc['AVG', 'OOD-Sum'] = cost_df.loc[main_models, 'OOD-Sum'].mean(skipna=True)
                cost_df.loc['Max Expert', 'OOD-Sum'] = cost_df.loc[main_models, 'OOD-Sum'].max(skipna=True)

            if 'Random Router' in cost_df.index and ood_keys:
                ood_random_costs = {k: v for k, v in random_router_costs.items() if k in ood_keys}
                if cost_metric == 'total_cost':
                    cost_df.loc['Random Router', 'OOD-Sum'] = sum(ood_random_costs.values())
                else:
                    cost_df.loc['Random Router', 'OOD-Sum'] = sum(ood_random_costs.values()) / len(ood_random_costs) if ood_random_costs else None

            if 'Oracle' in cost_df.index and ood_keys:
                ood_oracle_costs = {k: v for k, v in oracle_costs.items() if k in ood_keys}
                if cost_metric == 'total_cost':
                    cost_df.loc['Oracle', 'OOD-Sum'] = sum(ood_oracle_costs.values())
                else:
                    cost_df.loc['Oracle', 'OOD-Sum'] = sum(ood_oracle_costs.values()) / len(ood_oracle_costs) if ood_oracle_costs else None

        # Add Dataset-Avg column for performance (average across all datasets for each model)
        # First compute for actual models
        for model in main_models + ref_models:
            if model in performance_df.index:
                performance_df.loc[model, 'Dataset-Avg'] = performance_df.loc[model, ordered_columns].mean(skipna=True)

        # Now compute for aggregate rows
        if main_models:
            # AVG: average of main models' Dataset-Avg values
            performance_df.loc['AVG', 'Dataset-Avg'] = performance_df.loc[main_models, 'Dataset-Avg'].mean(skipna=True)
            # Max Expert: Dataset-Avg is the mean of its per-dataset maxima
            performance_df.loc['Max Expert', 'Dataset-Avg'] = performance_df.loc['Max Expert', ordered_columns].mean(skipna=True)

        # Random Router: compute dataset-level average (not sample-level)
        if 'Random Router' in performance_df.index:
            all_random_values = [random_router_stats.get(key) for key in ordered_columns if random_router_stats.get(key) is not None]
            performance_df.loc['Random Router', 'Dataset-Avg'] = sum(all_random_values) / len(all_random_values) if all_random_values else None

        # Oracle: compute dataset-level average (not sample-level)
        if 'Oracle' in performance_df.index:
            all_oracle_values = [oracle_stats.get(key) for key in ordered_columns if oracle_stats.get(key) is not None]
            performance_df.loc['Oracle', 'Dataset-Avg'] = sum(all_oracle_values) / len(all_oracle_values) if all_oracle_values else None

        # Add Dataset-Sum column for cost (sum across all datasets for each model)
        # First compute for actual models
        for model in main_models + ref_models:
            if model in cost_df.index:
                cost_df.loc[model, 'Dataset-Sum'] = cost_df.loc[model, ordered_columns].sum(skipna=True)

        # Now compute for aggregate rows
        if main_models:
            cost_df.loc['AVG', 'Dataset-Sum'] = cost_df.loc[main_models, 'Dataset-Sum'].mean(skipna=True)
            cost_df.loc['Max Expert', 'Dataset-Sum'] = cost_df.loc[main_models, 'Dataset-Sum'].max(skipna=True)

        if 'Random Router' in cost_df.index:
            if cost_metric == 'total_cost':
                cost_df.loc['Random Router', 'Dataset-Sum'] = sum(random_router_costs.values())
            else:
                cost_df.loc['Random Router', 'Dataset-Sum'] = sum(random_router_costs.values()) / len(random_router_costs) if random_router_costs else None

        if 'Oracle' in cost_df.index:
            if cost_metric == 'total_cost':
                cost_df.loc['Oracle', 'Dataset-Sum'] = sum(oracle_costs.values())
            else:
                cost_df.loc['Oracle', 'Dataset-Sum'] = sum(oracle_costs.values()) / len(oracle_costs) if oracle_costs else None

        # Add Sample-Avg column (weighted by sample count across all datasets)
        performance_df['Sample-Avg'] = performance_df.index.map(lambda x: sample_level_avg_dict.get(x))

        # Reorder rows: main models -> aggregate rows -> reference models
        aggregate_rows = ['AVG', 'Random Router', 'Max Expert', 'Oracle']

        # Determine final row order
        ordered_rows = []
        # Add main models (sorted alphabetically)
        ordered_rows.extend(sorted(main_models))
        # Add aggregate rows
        for agg_row in aggregate_rows:
            if agg_row in performance_df.index:
                ordered_rows.append(agg_row)
        # Add reference models (sorted alphabetically)
        ordered_rows.extend(sorted(ref_models))

        # Reindex both DataFrames
        performance_df = performance_df.reindex(ordered_rows)
        cost_df = cost_df.reindex(ordered_rows)

        # Fill missing values with "-" for display
        performance_df = performance_df.fillna("-")
        cost_df = cost_df.fillna("-")

        return performance_df, cost_df

    def print_summary_tables(self,
                            cost_metric: str = 'total_cost',
                            score_as_percent: bool = True,
                            precision: int = 2,
                            test_mode: bool = False,
                            random_seed: int = 42,
                            train_ratio: float = 0.8,
                            ood_datasets: Optional[List[str]] = None) -> None:
        """
        Print performance and cost summary tables in a formatted view.

        This is a convenience method that calls to_summary_table() and prints
        the results in a human-readable format.

        Args:
            cost_metric: Cost metric to use ('total_cost' or 'avg_cost_per_record')
            score_as_percent: Display scores as percentages (e.g., 85.00 instead of 0.85)
                             Default is True (percentage format)
            precision: Number of decimal places for numerical values (default: 2)
            test_mode: If True, only compute statistics for test set (requires data_loader)
            random_seed: Random seed for:
                        - Random Router sampling (for each prompt, randomly select one model)
                        - Train/test splitting (used when test_mode=True)
            train_ratio: Proportion of data for training (used when test_mode=True)
            ood_datasets: List of dataset IDs to treat as OOD/test only (used when test_mode=True)

        Note:
            Reference models are loaded from data_loader.reference_models (from config file).
            These models are excluded from AVG, Random Router, Max Expert, and Oracle calculations,
            and are displayed at the bottom for comparison only.

        Example:
            >>> agg = BaselineAggregator(records)
            >>> # Default: decimal format (without OOD datasets)
            >>> agg.print_summary_tables()
            ========== Performance Table (avg_score) ==========
                                aime/hybrid  humaneval/test  bbh/test  Dataset-Avg  Sample-Avg
            gpt-4                       0.85            0.92      0.78         0.85        0.85
            claude-3                    0.82            0.89         -         0.86        0.85
            llama-3                     0.75            0.85      0.70         0.77        0.76
            AVG                         0.81            0.89      0.74         0.82        0.82
            Random Router               0.81            0.89      0.74         0.82        0.83
            Max Expert                  0.85            0.92      0.78         0.85        0.85
            Oracle                      0.90            0.95      0.82         0.89        0.89

            ========== Cost Table (total_cost) ==========
                                aime/hybrid  humaneval/test  bbh/test  Dataset-Sum
            gpt-4                       1.23            0.45      0.67         2.35
            claude-3                    0.98            0.38         -         1.36
            llama-3                     0.15            0.08      0.12         0.35
            AVG                         0.79            0.30      0.40         1.49
            Random Router               0.79            0.30      0.40         1.49
            Max Expert                  1.23            0.45      0.67         2.35
            Oracle                      0.80            0.32      0.42         1.54
            >>>
            >>> # With OOD datasets: columns reordered and separate averages computed
            >>> agg.print_summary_tables(ood_datasets=["bbh"])
            ========== Performance Table (avg_score) ==========
                        aime/hybrid  humaneval/test  bbh/test  In-Dist-Avg  OOD-Avg  Dataset-Avg  In-Dist-Sample-Avg  OOD-Sample-Avg  Sample-Avg
            gpt-4              0.85            0.92      0.78         0.885     0.78         0.85                0.89            0.78        0.85
            claude-3           0.82            0.89         -         0.855        -         0.86                0.85               -        0.85
            llama-3            0.75            0.85      0.70         0.800     0.70         0.77                0.78            0.70        0.76
            ...
            >>>
            >>> # Optional: use percentage format
            >>> agg.print_summary_tables(score_as_percent=True)
            ========== Performance Table (avg_score) ==========
                                aime/hybrid  humaneval/test  bbh/test  Dataset-Avg  Sample-Avg
            gpt-4                     85.00          92.00    78.00       85.00      85.00
            ...
            >>>
            >>> # Test mode: only show test set results
            >>> agg.print_summary_tables(
            ...     test_mode=True,
            ...     random_seed=42,
            ...     train_ratio=0.8
            ... )
            >>>
            >>> # Reference models are loaded from config file (data_loader.reference_models)
            >>> # These are excluded from AVG/Random Router/Max Expert/Oracle
            >>> # Config: filters.reference_models = ["gpt-4o", "claude-opus"]
            >>> agg.print_summary_tables()
            ========== Performance Table (avg_score) ==========
                                aime/hybrid  humaneval/test  Dataset-Avg  Sample-Avg
            gpt-4                       0.85            0.92         0.89        0.85
            claude-3                    0.82            0.89         0.86        0.85
            llama-3                     0.75            0.85         0.80        0.76
            AVG                         0.81            0.89         0.85        0.82
            Random Router               0.81            0.89         0.85        0.83
            Max Expert                  0.85            0.92         0.89        0.85
            Oracle                      0.90            0.95         0.93        0.89
            gpt-4o                      0.92            0.95         0.94        0.92
            claude-opus                 0.90            0.93         0.92        0.91
        """
        # Get the tables (this will check for pandas dependency)
        perf_df, cost_df = self.to_summary_table(
            cost_metric=cost_metric,
            test_mode=test_mode,
            random_seed=random_seed,
            train_ratio=train_ratio,
            ood_datasets=ood_datasets
        )

        # Format tables using shared formatting methods
        perf_display = self._format_performance_table(perf_df, score_as_percent, precision)
        cost_display = self._format_cost_table(cost_df, precision)

        # Print tables
        print("=" * 50)
        print(f"Performance Table (avg_score)")
        print("=" * 50)
        print(perf_display.to_string())
        print()

        print("=" * 50)
        print(f"Cost Table ({cost_metric})")
        print("=" * 50)
        print(cost_display.to_string())
        print()

    def save_summary_tables_to_excel(self,
                                     output_file: str,
                                     cost_metric: str = 'total_cost',
                                     score_as_percent: bool = True,
                                     precision: int = 2,
                                     test_mode: bool = False,
                                     random_seed: int = 42,
                                     train_ratio: float = 0.8,
                                     ood_datasets: Optional[List[str]] = None) -> None:
        """
        Save performance and cost summary tables to an Excel file.

        This method creates an Excel file with two sheets:
        - Sheet 1 "Performance": Performance table (avg_score)
        - Sheet 2 "Cost ({cost_metric})": Cost table with specified metric

        The tables are formatted identically to print_summary_tables() output.

        Args:
            output_file: Path to the output Excel file (e.g., "summary.xlsx")
            cost_metric: Cost metric to use ('total_cost' or 'avg_cost_per_record')
            score_as_percent: Display scores as percentages (e.g., 85.00 instead of 0.85)
                             Default is True (percentage format)
            precision: Number of decimal places for numerical values (default: 2)
            test_mode: If True, only compute statistics for test set (requires data_loader)
            random_seed: Random seed for:
                        - Random Router sampling (for each prompt, randomly select one model)
                        - Train/test splitting (used when test_mode=True)
            train_ratio: Proportion of data for training (used when test_mode=True)
            ood_datasets: List of dataset IDs to treat as OOD/test only (used when test_mode=True)

        Note:
            Reference models are loaded from data_loader.reference_models (from config file).
            These models are excluded from AVG, Random Router, Max Expert, and Oracle calculations,
            and are displayed at the bottom for comparison only.

        Raises:
            ImportError: If pandas or openpyxl is not installed

        Example:
            >>> agg = BaselineAggregator(records, data_loader=loader)
            >>> # Basic usage (without OOD datasets)
            >>> agg.save_summary_tables_to_excel('summary.xlsx')
            >>>
            >>> # With OOD datasets: columns reordered and separate averages computed
            >>> agg.save_summary_tables_to_excel(
            ...     'summary.xlsx',
            ...     ood_datasets=['brainteaser', 'dailydialog']
            ... )
            >>> # Excel will have:
            >>> # - Performance sheet with columns: [in-dist...] | [OOD...] | In-Dist-Avg | OOD-Avg |
            >>> #   Dataset-Avg | In-Dist-Sample-Avg | OOD-Sample-Avg | Sample-Avg
            >>> # - Cost sheet with columns: [in-dist...] | [OOD...] | In-Dist-Sum | OOD-Sum | Dataset-Sum
            >>>
            >>> # With percentage format
            >>> agg.save_summary_tables_to_excel('summary.xlsx', score_as_percent=True)
            >>>
            >>> # Test mode with custom settings
            >>> agg.save_summary_tables_to_excel(
            ...     'test_summary.xlsx',
            ...     test_mode=True,
            ...     random_seed=42,
            ...     train_ratio=0.7,
            ...     ood_datasets=['brainteaser']
            ... )
            >>>
            >>> # Use avg cost per record
            >>> agg.save_summary_tables_to_excel(
            ...     'summary.xlsx',
            ...     cost_metric='avg_cost_per_record',
            ...     precision=3
            ... )
        """
        # Check for openpyxl dependency
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is required for save_summary_tables_to_excel(). "
                "Install with: pip install openpyxl"
            )

        # Get the tables (this will check for pandas dependency)
        perf_df, cost_df = self.to_summary_table(
            cost_metric=cost_metric,
            test_mode=test_mode,
            random_seed=random_seed,
            train_ratio=train_ratio,
            ood_datasets=ood_datasets
        )

        # Format tables using shared formatting methods
        perf_display = self._format_performance_table(perf_df, score_as_percent, precision)
        cost_display = self._format_cost_table(cost_df, precision)

        # Import pandas (already verified by to_summary_table)
        import pandas as pd

        # Write to Excel with two sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write Performance sheet
            perf_display.to_excel(writer, sheet_name='Performance', index=True)

            # Write Cost sheet with dynamic name
            cost_sheet_name = f'Cost ({cost_metric})'
            cost_display.to_excel(writer, sheet_name=cost_sheet_name, index=True)

        logger.info(f"Summary tables saved to {output_file}")
        logger.info(f"  - Sheet 1: Performance (avg_score)")
        logger.info(f"  - Sheet 2: {cost_sheet_name}")
