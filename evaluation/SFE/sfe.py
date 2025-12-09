import os
import json
import re
from typing import Dict, Optional, Any

from datasets import Dataset, disable_progress_bars

from evaluation.SFE.prompts import (
    SFE_GRADER_TEMPLATE,
    MCQ_PROMPT,
    EXACT_MATCH_PROMPT,
    OPEN_QUESTION_PROMPT,
)
from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/SFE"

class SFEEvaluator(BaseEvaluator):
    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        super().__init__(grader_cache_config)
        self.task = "SFE"
        self.seed = 42
        
    def load_data(self, split: str = "test"):
        """Load SFE data from local JSONL file"""
        data_file = os.path.join(DATA_DIR, f"{split}.jsonl")
        
        data = self.load_jsonl(data_file)
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field for compatibility
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test"]

    def format_prompt(self, example: Dict) -> Dict:
        """Format the prompt for SFE questions with image tag replacement"""
        question = example["question"]
        question_type = example.get("question_type", "exact_match")
        options = example.get("options", [])
        images = example.get("images", [])
        discipline = example.get("field") or example.get("category") or "science"
        
        # Build prompt prefix based on question type, mirroring VLMEvalKit logic
        question_type_lower = question_type.lower()
        if question_type_lower == "mcq":
            prefix = MCQ_PROMPT.format(discipline=discipline)
        elif question_type_lower == "open_ended":
            prefix = OPEN_QUESTION_PROMPT.format(discipline=discipline)
        else:
            prefix = EXACT_MATCH_PROMPT.format(discipline=discipline)
        
        composed_question = f"{prefix} {question}".strip()

        if options:
            options_text = "\n".join(options)
            composed_question += f"\nChoices are:\n{options_text}"
        
        # Prepare image paths for the generator
        image_paths = []
        if images:
            for img_name in images:
                # Construct absolute path to image in SFE images directory
                img_path = os.path.abspath(os.path.join(DATA_DIR, "images", img_name))
                image_paths.append(img_path)
        
        return {
            **example,
            "prompt": composed_question,
            "image_paths": image_paths
        }
    
    def _get_grader_result(self, question: str, correct_answer: str, predicted_answer: str, question_type: str) -> Dict:
        """
        Get grading result from grader model with automatic caching and retry

        Returns:
            Dict with keys: score (0-10)
        """
        prompt = SFE_GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=predicted_answer
        )

        # Call grader (DirectGenerator) - automatic caching + retry
        result = self.grader.generate(prompt)

        # Parse the structured response
        return self._parse_grader_response(result.output)
    
    def _parse_grader_response(self, response_text: str) -> Dict:
        """Parse the grader response to extract the 0-10 score"""
        
        # Initialize default values
        result = {
            "score": 0
        }
        
        try:
            # Extract the numeric score (0-10)
            # Look for a single integer in the response
            score_match = re.search(r'\b(\d+)\b', response_text.strip())
            if score_match:
                score = int(score_match.group(1))
                # Ensure score is within valid range
                if 0 <= score <= 10:
                    result["score"] = score
                else:
                    print(f"Score {score} out of range, defaulting to 0")
                    result["score"] = 0
            else:
                print(f"No valid score found in response: {response_text}")
                result["score"] = 0
            
        except Exception as e:
            print(f"Error parsing grader response: {e}")
            result["score"] = 0
        
        return result
    
    def get_grader_result(self, question: str, answer: str, raw_data: str, question_type: str) -> Dict:
        """
        Public interface for getting grader result
        
        Args:
            question: The original question
            answer: The correct answer
            raw_data: The model's response to be evaluated
            question_type: Type of question (exact_match, mcq, open_ended)
            
        Returns:
            Dict with grading information
        """
        return self._get_grader_result(question=question, correct_answer=answer, 
                                     predicted_answer=raw_data, question_type=question_type)
    
    def extract_answer(self, prediction: str, question_type: str) -> str:
        """Extract answer from model prediction based on question type"""
        if question_type == "mcq":
            # For multiple choice, look for A, B, C, D patterns
            answer_patterns = [
                r"(?i)answer\s*:?\s*([A-Z])",
                r"(?i)^([A-Z])[.\s]",
                r"(?i)\b([A-Z])\b",
                r"\(([A-Z])\)",  # Match (A), (B), etc.
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, prediction.strip())
                if match:
                    return match.group(1).upper()
            
            # If no clear pattern found, return first letter that matches A-Z
            for char in prediction:
                if char.upper() in 'ABCDEFGHIJ':
                    return char.upper()
            
            return ""
        
        elif question_type == "exact_match":
            # For exact match, try to extract numeric values or clean text
            # First try to extract boxed content
            boxed = self.extract_boxed_content(prediction)
            if boxed:
                return boxed.strip()
            
            # Try to extract numbers for numeric answers
            number_match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', prediction)
            if number_match:
                return number_match.group().strip()
            
            # Clean up the prediction for text answers
            cleaned = re.sub(r'^(?:answer|response|result)\s*:?\s*', '', prediction.strip(), flags=re.IGNORECASE)
            return cleaned.strip()
        
        else:  # open_ended
            # For open-ended questions, clean up the response
            cleaned = re.sub(r'^(?:answer|response|result)\s*:?\s*', '', prediction.strip(), flags=re.IGNORECASE)
            return cleaned.strip()
    
    def evaluate(self, data, output_text, **kwargs):
        """
        Evaluate a single response using the third-party grader
        
        Args:
            data: Dictionary containing question and answer
            output_text: The model's response
            
        Returns:
            Dictionary with evaluation results matching other evaluators' format
        """
        answer = data['answer']
        question = data['prompt']
        question_type = data.get('question_type', 'exact_match')
        
        # Use grader to evaluate the answer
        grader_result = self.get_grader_result(
            question=question, 
            answer=answer, 
            raw_data=output_text,
            question_type=question_type
        )
        
        # Get score (0-10) and convert to decimal (0-1)
        score_0_10 = grader_result.get("score", 0)
        is_correct = score_0_10 / 10.0
        
        # Extract prediction from the raw model output
        prediction = self.extract_answer(output_text, question_type)
        
        # Return format consistent with other evaluators
        return {
            "prediction": prediction,
            "ground_truth": answer,
            "is_correct": is_correct
        }
