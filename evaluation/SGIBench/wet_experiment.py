"""
SGI-Bench Wet Experiment Evaluator

Task 3.2: Protocol Generation for wet lab experiments.
Evaluates experimental protocol generation using action sequence similarity
and parameter accuracy metrics.
"""

import os
from typing import Dict, Any, List, Optional
from datasets import Dataset

from evaluation.SGIBench.base import SGIBenchBaseEvaluator
from evaluation.SGIBench.prompts import WET_EXPERIMENT_OUTPUT_REQUIREMENTS
from evaluation.SGIBench.utils import (
    parse_experiment_steps,
    compare_exp_steps
)


class WetExperimentEvaluator(SGIBenchBaseEvaluator):
    """
    Evaluator for SGI-Bench Wet Experiment task.

    This task evaluates the model's ability to generate experimental protocols
    in a structured pseudocode format. Evaluation metrics include:
    - Action Sequence Similarity (Kendall tau distance)
    - Parameter Accuracy (step-by-step comparison)
    - Final Score (average of the above)
    """

    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize WetExperimentEvaluator.

        Args:
            grader_cache_config: Optional cache configuration for LLM grader
                                (not used in this evaluator as scoring is deterministic)
        """
        super().__init__(grader_cache_config)
        self.task = "SGIBench-WetExperiment"
        self.data_path = os.path.join(self.DATA_DIR, "wet_experiment", "test.jsonl")

    def load_data(self, split: str = "test") -> Dataset:
        """
        Load wet experiment data.

        Args:
            split: Data split (only "test" is supported)

        Returns:
            Dataset with formatted prompts
        """
        data = self.load_jsonl(self.data_path)
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.format_prompt)
        return dataset

    def get_valid_splits(self) -> List[str]:
        """Get valid data splits."""
        return ["test"]

    def format_prompt(self, item: Dict) -> Dict:
        """
        Format the prompt for wet experiment task.

        Appends output requirements to the original question.

        Args:
            item: Data item containing 'question' field

        Returns:
            Item with 'prompt' field added
        """
        question = item.get("question", "")
        prompt = question + WET_EXPERIMENT_OUTPUT_REQUIREMENTS

        return {
            **item,
            "prompt": prompt,
            "origin_query": question
        }

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        """
        Evaluate model output against ground truth protocol.

        Evaluation process:
        1. Extract protocol from model output (between <answer> tags)
        2. Parse experiment steps from both prediction and ground truth
        3. Compare using action sequence similarity and parameter accuracy

        Args:
            data: Input data item containing 'answer' (ground truth protocol)
            output_text: Model output text

        Returns:
            Evaluation result dictionary with:
            - prediction: Extracted protocol from model output
            - ground_truth: Ground truth protocol
            - is_correct: Boolean indicating if final_score >= 0.8
            - action_sequence_similarity: Kendall tau distance score
            - parameter_accuracy: Parameter comparison accuracy
            - final_score: Average of the two metrics
            - comparison_details: Detailed step-by-step comparison
        """
        # Extract protocol from model output
        extracted_answer = self.extract_sgi_answer(output_text)

        if extracted_answer is None:
            # Fallback: use the entire output
            extracted_answer = output_text

        # Get ground truth
        ground_truth = data.get("answer", "")

        # Parse experiment steps
        gt_steps = parse_experiment_steps(ground_truth)
        pred_steps = parse_experiment_steps(extracted_answer)

        # Handle edge cases
        if not gt_steps or not pred_steps:
            return {
                "prediction": extracted_answer,
                "ground_truth": ground_truth,
                "is_correct": False,
                "action_sequence_similarity": 0.0,
                "parameter_accuracy": 0.0,
                "final_score": 0.0,
                "comparison_details": [],
                "error": "Failed to parse experiment steps"
            }

        # Compare steps
        comparison_result = compare_exp_steps(gt_steps, pred_steps)

        # Calculate final score
        action_seq_sim = comparison_result["order_similarity"]
        param_acc = comparison_result["parameter_acc"]
        final_score = (action_seq_sim + param_acc) / 2

        return {
            "prediction": extracted_answer,
            "ground_truth": ground_truth,
            "is_correct": final_score,  # Use continuous score
            "action_sequence_similarity": action_seq_sim,
            "parameter_accuracy": param_acc,
            "final_score": final_score,
            "comparison_details": comparison_result["details"]
        }
