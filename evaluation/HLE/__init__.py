"""
HLE (Human-Like Evaluation) evaluation module for LLMRouterBench.

This module provides evaluation capabilities for HLE dataset using third-party grader models.
"""

from .hle import HLEEvaluator

__all__ = ["HLEEvaluator"]