import os
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.HumanEval.execution import check_correctness
from evaluation.HumanEval.utils import imports, sanitize

disable_progress_bars()

DATA_DIR = "data/HumanEval"

PROMPT = """You are an expert Python programmer, and here is your task:
{question}

Your code should pass these tests:
{test}
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class HumanEvalEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "HumanEval"
        self.seed = 42
        self.imports = imports
        
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"HumanEval.jsonl"))    
        data = Dataset.from_list(data)
    
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('prompt', '')})
        
        data = data.map(lambda x: self.format_prompt(x))
        
            
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict):
        # answer key: Answer
        prompt = PROMPT.format(
            question=item["prompt"],
            test=item["test"]
        )
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str, test: str, entry_point: str) -> str:
        answer = self.extract_code_answer(text=raw_data, test=test, entry_point=entry_point)
        if answer is None:
            answer = ""
        return answer
    
    def extract_code_answer(self, text: str, test: str, entry_point: str) -> str:
        extract_code = sanitize(text)
        code = "\n".join(self.imports) + "\n" + extract_code + "\n" + test + "\n" + f"check({entry_point})"
        
        return code
    
    def evaluate(self, data, output_text, **kwargs):
        prediction = self.extract_raw_answer(raw_data=output_text, test=data['test'], entry_point=data['entry_point'])
        
        is_correct = check_correctness(task_id=data['task_id'], completion_id=0, solution=prediction, time_out=10)['passed']
        
        return {
            "prediction": prediction,
            "ground_truth": data.get('canonical_solution', None),
            "is_correct": is_correct
        }
    
