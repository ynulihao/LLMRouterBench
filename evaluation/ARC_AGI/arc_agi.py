import os
import json
import re
from typing import Dict, List, Any, Optional
from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.ARC_AGI.prompts import ARC_AGI_PROMPT_TEMPLATE, format_grid

disable_progress_bars()

DATA_DIR = "data/ARC_AGI"


class ARCAGIEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "ARC-AGI"
        self.seed = 42
        
    def load_data(self, split: str = "v1"):
        """Load ARC-AGI data from JSONL file - Task-level evaluation (400 tasks)"""
        data_file = os.path.join(DATA_DIR, f"{split}.jsonl")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"ARC-AGI data file not found: {data_file}\n"
                f"Expected file: {split}.jsonl"
            )
        
        data = self.load_jsonl(data_file)
        
        # Process each task as a single evaluation unit (400 tasks total)
        processed_data = []
        for task in data:
            task_id = task.get("id", "unknown")
            train_examples = task.get("train", [])
            test_examples = task.get("test", [])
            
            # Each task is one evaluation unit, regardless of number of test examples
            sample = {
                "task_id": task_id,
                "train_examples": train_examples,
                "test_examples": test_examples,  # All test examples for this task
                "total_test_count": len(test_examples),
                "id": task_id  # Task ID as unique identifier
            }
            processed_data.append(sample)
        
        data = Dataset.from_list(processed_data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field for compatibility
        data = data.map(lambda x: {**x, 'origin_query': f"ARC-AGI Task {x.get('task_id', '')}"})
        
        return data
    
    def get_valid_splits(self):
        return ["v1"]
    
    def format_prompt(self, item: Dict) -> Dict:
        """Format the prompt for ARC-AGI tasks"""
        train_examples = item["train_examples"]
        test_examples = item["test_examples"]
        
        # Format training examples
        formatted_examples = []
        for i, example in enumerate(train_examples, 1):
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])
            
            example_text = f"""Example {i}:

Input:
{format_grid(input_grid)}

Output:
{format_grid(output_grid)}"""
            
            formatted_examples.append(example_text)
        
        training_examples_text = "\n\n".join(formatted_examples)
        
        # Format test inputs (may be multiple)
        if len(test_examples) == 1:
            # Single test case
            test_input_text = format_grid(test_examples[0].get("input", []))
        else:
            # Multiple test cases - format all inputs
            test_inputs = []
            for i, test_example in enumerate(test_examples, 1):
                test_input = test_example.get("input", [])
                test_inputs.append(f"Test Input {i}:\n{format_grid(test_input)}")
            test_input_text = "\n\n".join(test_inputs)
        
        prompt = ARC_AGI_PROMPT_TEMPLATE.format(
            training_examples=training_examples_text,
            test_input=test_input_text
        )
        
        return {"prompt": prompt}
    
    def parse_output(self, output_text: str) -> Optional[List[List[int]]]:
        """
        Parse model output to extract the predicted grid.
        Supports JSON array format.
        """
        if not output_text:
            return None
        
        # Method 1: Backscan for JSON arrays (most reliable for complete arrays)
        last_bracket_idx = -1
        for i in range(len(output_text) - 1, -1, -1):
            if output_text[i] == ']':
                last_bracket_idx = i
                break
        
        if last_bracket_idx != -1:
            bracket_count = 1
            start_idx = -1
            
            for i in range(last_bracket_idx - 1, -1, -1):
                char = output_text[i]
                if char == ']':
                    bracket_count += 1
                elif char == '[':
                    bracket_count -= 1
                    if bracket_count == 0:
                        start_idx = i
                        break
            
            if start_idx != -1:
                json_candidate = output_text[start_idx:last_bracket_idx + 1]
                try:
                    parsed = json.loads(json_candidate)
                    if self._is_valid_grid(parsed):
                        return parsed
                except json.JSONDecodeError:
                    pass
        
        # Method 2: Look for JSON array pattern with improved regex
        json_pattern = r'\[\s*\[[\d\s,\[\]]*\]\s*\]'
        matches = re.findall(json_pattern, output_text, re.DOTALL)
        
        for match in reversed(matches):  # Try from the end first
            try:
                parsed = json.loads(match)
                if self._is_valid_grid(parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _is_valid_grid(self, grid: Any) -> bool:
        """Check if the parsed output is a valid grid format"""
        if not isinstance(grid, list):
            return False
        
        if not grid:  # Empty grid
            return False
        
        # Check if all elements are lists (rows)
        if not all(isinstance(row, list) for row in grid):
            return False
        
        # Check if all rows have the same length
        if len(set(len(row) for row in grid)) > 1:
            return False
        
        # Check if all elements are integers
        for row in grid:
            if not all(isinstance(cell, int) for cell in row):
                return False
        
        return True
    
    def grid_exact_match(self, predicted: List[List[int]], ground_truth: List[List[int]]) -> bool:
        """Check if two grids are exactly equal"""
        if len(predicted) != len(ground_truth):
            return False
        
        for pred_row, gt_row in zip(predicted, ground_truth):
            if len(pred_row) != len(gt_row):
                return False
            if pred_row != gt_row:
                return False
        
        return True
    
    
    def parse_multiple_outputs(self, output_text: str, num_expected: int) -> List[Optional[List[List[int]]]]:
        """Parse model output for multiple test cases"""
        if num_expected == 1:
            # Single output case
            parsed = self.parse_output(output_text)
            return [parsed]
        
        # Improved pattern to match complete JSON arrays (including multi-line)
        # This pattern matches opening [ followed by content including nested arrays, then closing ]
        json_pattern = r'\[(?:\s*\[[\d\s,]*\]\s*,?\s*)*\]'
        matches = re.findall(json_pattern, output_text, re.DOTALL)
        
        parsed_outputs = []
        for match in matches:
            # Clean up the match (remove extra whitespace)
            cleaned_match = re.sub(r'\s+', ' ', match.strip())
            try:
                parsed = json.loads(cleaned_match)
                if self._is_valid_grid(parsed):
                    parsed_outputs.append(parsed)
                # Skip invalid grids instead of adding None immediately
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue
        
        # If we didn't find enough valid arrays, try alternative approach
        if len(parsed_outputs) < num_expected:
            # Split by double newlines (our suggested format) and try parsing each part
            parts = output_text.split('\n\n')
            for part in parts:
                if len(parsed_outputs) >= num_expected:
                    break
                    
                part = part.strip()
                if part and part.startswith('[') and part.endswith(']'):
                    try:
                        parsed = json.loads(part)
                        if self._is_valid_grid(parsed):
                            parsed_outputs.append(parsed)
                    except json.JSONDecodeError:
                        continue
        
        # Ensure we have the right number of outputs (pad with None if needed)
        while len(parsed_outputs) < num_expected:
            parsed_outputs.append(None)
        
        return parsed_outputs[:num_expected]

    def evaluate(self, data, output_text, **kwargs):
        """
        Evaluate a single ARC-AGI task (may contain multiple test cases)
        
        Args:
            data: Dictionary containing task information
            output_text: The model's response
            
        Returns:
            Dictionary with evaluation results in standard format
        """
        test_examples = data.get('test_examples', [])
        total_test_count = len(test_examples)
        
        # Parse the model output(s)
        predicted_grids = self.parse_multiple_outputs(output_text, total_test_count)
        
        # Collect predictions and ground truths
        predictions = []
        ground_truths = []
        all_correct = True
        
        for test_example, predicted_grid in zip(test_examples, predicted_grids):
            ground_truth = test_example.get("output", [])
            
            predictions.append(predicted_grid)
            ground_truths.append(ground_truth)
            
            # Check if this test case is correct
            if predicted_grid is None:
                all_correct = False
            else:
                is_exact_match = self.grid_exact_match(predicted_grid, ground_truth)
                if not is_exact_match:
                    all_correct = False
        
        # Return standard format - task passes only if ALL test cases are correct
        return {
            "prediction": predictions,
            "ground_truth": ground_truths,
            "is_correct": all_correct
        }