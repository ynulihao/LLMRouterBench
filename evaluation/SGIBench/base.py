"""
SGI-Bench Base Evaluator

Shared base class for all SGI-Bench subtasks.
"""

import os
import re
from typing import Optional, Dict, Any, List
from datasets import Dataset

from evaluation.base_evaluator import BaseEvaluator


class SGIBenchBaseEvaluator(BaseEvaluator):
    """Base class for all SGI-Bench subtasks"""

    # All 10 disciplines in SGI-Bench
    DISCIPLINES = [
        'astronomy', 'chemistry', 'earth', 'energy', 'information',
        'life', 'material', 'mathematics', 'neuroscience', 'physics'
    ]

    # Data directory
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sgibench")

    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize SGI-Bench evaluator.

        Args:
            grader_cache_config: Optional cache configuration for LLM grader
        """
        super().__init__(grader_cache_config)
        self._grader_cache_config = grader_cache_config

    def _init_sgi_grader(self, model_env_var: str, default_model: str, temperature: float = 0.0):
        """
        Initialize SGI-Bench specific grader.

        Subclasses should call this method to override the default grader with
        task-specific configuration aligned with official SGI-Bench implementation.

        Args:
            model_env_var: Environment variable name for model (e.g., "SGI_DR_GRADER_MODEL")
            default_model: Default model name if env var not set (e.g., "o4-mini")
            temperature: Temperature for generation (default: 0.0)
        """
        from generators.generator import DirectGenerator

        model_name = os.getenv(model_env_var, default_model)
        base_url = os.getenv("GRADER_BASE_URL", "")
        api_key = os.getenv("GRADER_API_KEY", "")

        self.grader = DirectGenerator(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=500,
            cache_config=self._grader_cache_config
        )

    def extract_sgi_answer(self, text: str, start_tag: str = '<answer>', end_tag: str = '</answer>') -> Optional[str]:
        """
        Extract content between answer tags.

        This is the official SGI-Bench answer extraction logic from utils.py.
        Uses rfind to get the LAST occurrence of the start tag.

        Args:
            text: The text to extract answer from
            start_tag: Start tag (default: '<answer>')
            end_tag: End tag (default: '</answer>')

        Returns:
            Extracted content or None if not found
        """
        if text is None:
            return None
        text = str(text)

        # Find the last occurrence of start_tag
        start_index = text.rfind(start_tag)
        if start_index != -1:
            end_index = text.find(end_tag, start_index)
            if end_index != -1:
                return text[start_index + len(start_tag):end_index].strip()
        return None

    def load_data(self, split: str = "test") -> Dataset:
        """
        Load data for the specific task.

        Subclasses should override this method to specify the correct data path.

        Args:
            split: Data split (default: "test")

        Returns:
            Dataset with formatted prompts
        """
        raise NotImplementedError("Subclasses must implement load_data()")

    def get_valid_splits(self) -> List[str]:
        """
        Get valid data splits for this evaluator.

        Returns:
            List of valid split names
        """
        return ["test"]

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        """
        Evaluate model output against ground truth.

        Subclasses must implement this method with task-specific evaluation logic.

        Args:
            data: Input data item
            output_text: Model output text
            **kwargs: Additional arguments

        Returns:
            Evaluation result dictionary
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def format_prompt(self, item: Dict) -> Dict:
        """
        Format the prompt for model input.

        Subclasses should override this to add task-specific prompt formatting.

        Args:
            item: Data item

        Returns:
            Item with formatted prompt
        """
        return item
