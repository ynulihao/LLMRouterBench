import os
from typing import Dict

from datasets import Dataset, disable_progress_bars
from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/dailydialog"

PROMPT = """Given a conversation history and a current utterance, follow these steps to identify the emotion of the current utterance from the given options. The emotion should be determined based on both the conversation context and the current utterance.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFG. Let's think step by step.

History:
{history}

Utterance:
{utterance}

Options:
{options}"""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-G])[.\s\n]?"

class DailyDialogEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "DailyDialog"
        self.seed = 42
        
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        data = data.map(lambda x: {**x, 'origin_query': x.get('utterance', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            history = "- "+"\n- ".join(item["history"]),
            utterance = item["utterance"],
            options = "\n".join([f"{key}. {value}" for key, value in item["candidate"].items()])
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
    
