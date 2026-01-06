"""
LLMRouterBench Baseline Data Loading System

This package provides efficient loading, transformation, and analysis of
benchmark results for baseline comparisons across datasets and models.

Main Components:
- BaselineDataLoader: Load and transform benchmark results
- BaselineRecord: Unified schema for benchmark records
- BaselineAggregator: Compute statistics and comparisons
- AggregatedStats: Summary statistics for dataset/model combinations

Quick Start:
    >>> from baselines import BaselineDataLoader, BaselineAggregator
    >>> loader = BaselineDataLoader('config/baseline_config.yaml')
    >>> records = loader.load_all_records()

    >>> # Compute aggregated statistics
    >>> aggregator = BaselineAggregator(records)
    >>> perf_table, cost_table = aggregator.to_summary_table()
    >>> print(perf_table)
"""

from .data_loader import BaselineDataLoader
from .schema import BaselineRecord, AggregatedStats
from .aggregators import BaselineAggregator

__version__ = '1.0.0'

__all__ = [
    'BaselineDataLoader',
    'BaselineRecord',
    'AggregatedStats',
    'BaselineAggregator',
]
