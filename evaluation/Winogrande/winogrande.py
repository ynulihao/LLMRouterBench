import os
from collections import Counter
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/winogrande"

PROMPT = """You are given a *cloze* sentence containing a blank marked by underscores `___`. Only **one** of the two options correctly fills the blank while preserving commonsense.

After your reasoning, end with a line **exactly** in the form:  `Answer: $LETTER`, where `LETTER` is **A** or **B**.

Cloze Sentence:
{sentence}

Options:
A. {option1}
B. {option2}

Let's think step by step."""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-B])[.\s\n]?"

class WinograndeEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "Winogrande"
        self.seed = 42
    
    def load_data(self, split: str = "vaild"):
        # Winogrande only has valid split available
        data = self.load_jsonl(os.path.join(DATA_DIR, f"valid.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('sentence', '') + " " + x.get('option1', '') + " " + x.get('option2', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["valid"]
    
    def format_prompt(self, item: Dict) -> Dict:
        answer = item['answer']
        if answer == "2":
            project_answer = "B"
        elif answer == "1":
            project_answer = "A"
        else:
            raise ValueError(f"Invalid answer: {answer}")
        
        prompt = PROMPT.format(
            sentence = item["sentence"],
            option1 = item["option1"],
            option2 = item["option2"]
        )
        return {"prompt": prompt, "project_answer": project_answer}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        return self.extract_normal_answer(text=raw_data, answer_pattern=ANSWER_PATTERN)
    
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['project_answer']
        
        prediction = self.extract_raw_answer(raw_data=output_text)
        
        is_correct = answer == prediction
        
        return {
            "prediction": prediction,
            "ground_truth": answer,
            "is_correct": is_correct
        }
    