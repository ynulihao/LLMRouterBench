import os
from typing import Dict, List

from datasets import Dataset, disable_progress_bars
from evaluation.base_evaluator import BaseEvaluator
from evaluation.StudentEval.execution import check_correctness
from evaluation.StudentEval.utils import imports, sanitize

disable_progress_bars()

DATA_DIR = "data/studenteval"

PROMPT = """You are an expert Python programmer, and here is your task:
{question}

Your code should pass these tests:
{test}
""".strip()


class StudentEvalEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "StudentEval"
        self.seed = 42
        self.imports = imports
        
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        data = data.map(lambda x: {**x, 'origin_query': x.get('text', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            question=item['text'],
            test="\n".join(item['test_list'])
        )
        return {"task_prompt": prompt}
    
    def extract_code_answer(self, text: str, test_list: List[str]) -> str:
        extract_code = sanitize(text)
        code = "\n".join(self.imports) + "\n" + extract_code + "\n" + "\n".join(test_list)
        
        return code
    
    def extract_raw_answer(self, text: str, test_list: List[str]) -> str:
        answer = self.extract_code_answer(text=text, test_list=test_list)
        if answer is None:
            answer = ""
        return answer
    
    def evaluate(self, data, output_text, **kwargs):
        prediction = self.extract_raw_answer(text=output_text, test_list=data['test_list'])
        
        # Use check_correctness to validate the code
        task_id = kwargs.get('task_id', 0)
        is_correct = check_correctness(task_id=task_id, completion_id=0, solution=prediction, time_out=10)['passed']
        
        return {
            "prediction": prediction,
            "ground_truth": None,  # StudentEval doesn't have explicit ground truth answers
            "is_correct": is_correct
        }
    
