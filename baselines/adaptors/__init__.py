"""
Adaptors for converting LLMRouterBench baseline data to various baseline method formats.

This module provides data transformation utilities for:
- EmbedLLM: Matrix factorization-based routing method
- RouterDC: Deep learning-based router with clustering
- AvengersPro: Cluster-based routing method
- GraphRouter: Graph neural network-based routing method
- HybridLLM: Pairwise model routing with single sampling (Hybrid LLM)
- Model-SAT: Capability-instruction-based dynamic routing for task-aware LLM selection.
- FrugalGPT: A collection of techniques for building LLM applications with budget constraints
- RouteLLM: Matrix Factorization router fine-tuning assets
"""

from .common import split_by_dataset
from .embedllm_adaptor import EmbedLLMAdaptor
from .routerdc_adaptor import RouterDCAdaptor
from .avengerspro_adaptor import AvengersProAdaptor
from .graphrouter_adaptor import GraphRouterAdaptor
from .hybridllm_adaptor import HybridLLMAdaptor
from .modelsat_adaptor import ModelSATAdaptor
from .frugalgpt_adaptor import FrugalGPTAdaptor
from .routellm_adaptor import RouteLLMAdaptor
from .routerembedding_adaptor import RouterEmbeddingAdaptor

__all__ = [
    'split_by_dataset',
    'EmbedLLMAdaptor',
    'RouterDCAdaptor',
    'AvengersProAdaptor',
    'GraphRouterAdaptor',
    'HybridLLMAdaptor',
    'ModelSATAdaptor',
    'FrugalGPTAdaptor',
    'RouteLLMAdaptor',
    'RouterEmbeddingAdaptor',
]
