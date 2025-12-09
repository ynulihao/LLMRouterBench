import os
from typing import Dict

from datasets import Dataset, disable_progress_bars
from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/TruthfulQA"

PROMPT = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {candidates}.

{question}

{options}

Let's think step by step."""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?"

class TruthfulQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "TruthfulQA"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict) -> Dict:
        question = item["question"]
        options = item["options"]
        options = "\n".join([f"{chr(65+i)}) {option}" for i, option in enumerate(options)])
        prompt = PROMPT.format(
            question = question,
            candidates = "A, B, C, D",
            options = options
        )
        return {"prompt": prompt}
    
    def extract_raw_answer(self, text: str) -> str:
        return self.extract_normal_answer(text=text, answer_pattern=ANSWER_PATTERN)
    
    def evaluate(self, data, output_text, **kwargs):
        prediction = self.extract_raw_answer(text=output_text)
        ground_truth = data['answer']
        is_correct = prediction == ground_truth
        
        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        }
    
