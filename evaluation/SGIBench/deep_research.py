"""
SGI-Bench Deep Research Evaluator

Task 1: Deep Research - Multi-hop scientific question answering.
Evaluates using Exact Match and Step-Level Accuracy via LLM judge.
"""

import os
import re
from typing import Dict, Any, List, Optional

from datasets import Dataset
from loguru import logger

from evaluation.SGIBench.base import SGIBenchBaseEvaluator
from evaluation.SGIBench.prompts import (
    DEEP_RESEARCH_OUTPUT_REQUIREMENTS,
    DEEP_RESEARCH_JUDGE_PROMPT
)
from evaluation.SGIBench.utils import AnswerParser


class DeepResearchEvaluator(SGIBenchBaseEvaluator):
    """
    Evaluator for SGI-Bench Deep Research task.

    This task evaluates the model's ability to perform multi-hop scientific
    reasoning by answering questions that require information retrieval
    and calculation. Evaluation metrics include:
    - Exact Match (EM): Strict string equality with LLM parser normalization
    - Step-Level Accuracy (SLA): LLM-judged step-by-step correctness

    Per official implementation (step_2_score.py line 77), exact match checks:
    1. answer == model_answer (raw extracted answer)
    2. answer == model_answer_after_llm_paser (LLM-normalized answer)
    """

    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepResearchEvaluator.

        Args:
            grader_cache_config: Optional cache configuration for LLM grader
        """
        super().__init__(grader_cache_config)
        self.task = "SGIBench-DeepResearch"
        self.data_path = os.path.join(self.DATA_DIR, "deep_research", "test.jsonl")

        # Initialize grader with SGI-Bench Deep Research specific config
        # Official: judge = LLM('o4-mini') with temperature=0
        self._init_sgi_grader(
            model_env_var="SGI_DR_GRADER_MODEL",
            default_model="o4-mini",
            temperature=0.0
        )

        # Initialize answer parser for normalization (official: AnswerPaser in utils.py)
        self.answer_parser = AnswerParser(generator=self.grader)

    def load_data(self, split: str = "test") -> Dataset:
        """
        Load deep research data.

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
        Format the prompt for deep research task.

        Appends output requirements to the original question.

        Args:
            item: Data item containing 'question' field

        Returns:
            Item with 'prompt' field added
        """
        question = item.get("question", "")
        prompt = question + DEEP_RESEARCH_OUTPUT_REQUIREMENTS

        return {
            **item,
            "prompt": prompt,
            "origin_query": question
        }

    def _parse_step_level_accuracy(self, judge_output: str) -> float:
        """
        Parse step-level accuracy from LLM judge output.

        The judge output should be a JSON list of step evaluations.

        Args:
            judge_output: LLM judge output text

        Returns:
            Step-level accuracy score (0.0 to 1.0)
        """
        try:
            # Find the JSON list in the output
            start_index = judge_output.find('[')
            end_index = judge_output.rfind(']') + 1

            if start_index == -1 or end_index == 0:
                logger.warning("Could not find JSON list in judge output")
                return 0.0

            json_str = judge_output[start_index:end_index]

            # Try to parse JSON (with json_repair if available)
            try:
                from json_repair import repair_json
                steps = eval(repair_json(json_str))
            except ImportError:
                import json
                steps = json.loads(json_str)

            if not steps:
                return 0.0

            # Count correct steps
            correct_count = sum(
                1 for step in steps
                if isinstance(step, dict) and step.get("judge", "").lower() == "correct"
            )

            return correct_count / len(steps)

        except Exception as e:
            logger.warning(f"Failed to parse step-level accuracy: {e}")
            return 0.0


    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        """
        Evaluate model output against ground truth.

        Evaluation process (aligned with official step_2_score.py):
        1. Extract answer from model output (between <answer> tags) -> model_answer
        2. Normalize answer using LLM parser -> model_answer_after_llm_parser
        3. Calculate Exact Match: answer == model_answer OR answer == model_answer_after_llm_parser
        4. Use LLM judge to evaluate step-by-step reasoning

        Args:
            data: Input data item containing 'answer', 'question', 'steps'
            output_text: Model output text

        Returns:
            Evaluation result dictionary with:
            - prediction: Extracted answer from model output
            - ground_truth: Ground truth answer
            - is_correct: Boolean based on exact match
            - exact_match: 1 if answer matches, 0 otherwise
            - step_level_acc: Step-level accuracy from LLM judge
        """
        # Get ground truth
        ground_truth = data.get("answer", "")

        # Extract answer from model output (official: model_answer)
        extracted_answer = self.extract_sgi_answer(output_text)
        if extracted_answer is None:
            extracted_answer = output_text

        # Normalize answer using LLM parser (official: model_answer_after_llm_paser)
        # Per official step_1_get_answer.py lines 55-61:
        # - Use generic example "0.25" for numeric answers
        # - Use generic example "T cell and B cell" for text answers
        # IMPORTANT: Do NOT use ground_truth as example to avoid leaking the answer!
        parsed_answer = None
        try:
            # Determine example type based on ground_truth format (without leaking actual value)
            try:
                float(ground_truth)
                answer_example = "0.25"  # Generic numeric example
            except (ValueError, TypeError):
                answer_example = "T cell and B cell"  # Generic text example

            parsed_answer = self.answer_parser.parse(output_text, answer_example)
            if parsed_answer:
                logger.debug(f"Parsed answer: {parsed_answer}")
        except Exception as e:
            logger.warning(f"Answer parsing failed: {e}")

        # Calculate Exact Match (per official implementation line 77)
        # exact_match = 1 if (answer == model_answer or answer == model_answer_after_llm_paser) else 0
        exact_match = 0
        if ground_truth == extracted_answer:
            exact_match = 1
        elif parsed_answer is not None and ground_truth == parsed_answer:
            exact_match = 1

        # Prepare reference steps
        steps = data.get("steps", [])
        if isinstance(steps, list):
            reference_steps = '\n'.join(steps)
        else:
            reference_steps = str(steps)

        # Use LLM judge for step-level accuracy
        step_level_acc = 0.0
        llm_judge_output = None

        try:
            judge_prompt = DEEP_RESEARCH_JUDGE_PROMPT.format(
                question=data.get("question", ""),
                reference_steps=reference_steps,
                reference_answer=ground_truth,
                llm_solution=output_text,
                llm_answer=extracted_answer
            )

            judge_result = self.grader.generate(judge_prompt)
            llm_judge_output = judge_result.output

            step_level_acc = self._parse_step_level_accuracy(llm_judge_output)

            # Update token counts
            self.prompt_tokens += judge_result.prompt_tokens
            self.completion_tokens += judge_result.completion_tokens

        except Exception as e:
            logger.error(f"Failed to get LLM judge result: {e}")

        return {
            "prediction": extracted_answer,
            "parsed_answer": parsed_answer,
            "ground_truth": ground_truth,
            "is_correct": exact_match == 1,
            "exact_match": exact_match,
            "step_level_acc": step_level_acc,
            "llm_judge_output": llm_judge_output
        }
