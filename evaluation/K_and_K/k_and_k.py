import os
from typing import Dict

from datasets import Dataset, disable_progress_bars, load_dataset

from evaluation.base_evaluator import BaseEvaluator
from evaluation.K_and_K.scoring import parse_answer, judge_answer, ensemble_answers

disable_progress_bars()

DATA_DIR = "data/K_and_K"

PROMPT = """Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...

### Question:
{question}
"""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-D])[.\s\n]?"

class KnightsAndKnavesEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "Knights_and_Knaves"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        assert split in ["train", "test"]
        data = load_dataset(DATA_DIR, split=split)
        
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('quiz', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["train", "test"]
    
    def format_prompt(self, item: Dict) -> Dict:
        # format answer 
        answer = item["solution_text_format"].split("\n")
        answer = [a[3:].lstrip().strip() for a in answer if a.strip()]
        
        prompt = PROMPT.format(
            question = item["quiz"],
        )
        return {"prompt": prompt, "answer": answer}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        parsed_answer, is_success = parse_answer(pred_str=raw_data)
        return parsed_answer
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data["solution_text_format"].split("\n")
        answer = [a[3:].lstrip().strip() for a in answer if a.strip()]
        
        prediction = self.extract_raw_answer(raw_data=output_text)
        
        is_correct, wrong_reason, correct_ratio = judge_answer(
            pred_answer=prediction, reformat_gold_conditions=answer
        )
        
        return {
            "prediction": prediction,
            "ground_truth": answer,
            "is_correct": is_correct
        }
    
