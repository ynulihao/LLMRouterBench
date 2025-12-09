import os
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.deepscaler_rm import (extract_answer, grade_answer_mathd,
                                    grade_answer_sympy)

disable_progress_bars()


DATA_DIR = "data/bbh"


PROMPT = """Answer the logical question, Write a line of the form "Answer: $ANSWER" at the end of your response. Let's think step by step.

Here are 3 examples to help you understand how to answer the question:
{template}

Question: {question}
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class BBHEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "BBH"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('input', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict):
        prompt = PROMPT.format(
            template = item['fewshot_template'],
            question = item['input'],
        )
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        if "\\boxed" not in raw_data:
            answer = self.extract_normal_answer(text=raw_data, answer_pattern=ANSWER_PATTERN)
        else:
            answer = extract_answer(passage=raw_data)
        if answer is None:
            answer = ""
        return answer
    
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['target']
        
        prediction = self.extract_raw_answer(raw_data=output_text)
        
        if "\\boxed" in answer:
            ground_truth = extract_answer(passage=answer)
        else:
            ground_truth = answer
        
        if ground_truth is None or prediction is None or prediction == "":
            is_correct = False
        else:
            is_correct = grade_answer_mathd(prediction, ground_truth) \
                or grade_answer_sympy(prediction, ground_truth)
        
        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        }
    
