import os
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/arc_c"

PROMPT_FOUR_OPTIONS = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A) {A}
B) {B}
C) {C}
D) {D}

Let's think step by step."""

PROMPT_THREE_OPTIONS = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABC.

{question}

A) {A}
B) {B}
C) {C}

Let's think step by step."""

PROMPT_FIVE_OPTIONS = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

Let's think step by step."""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-E])[.\s\n]?"

class ARCCEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "ARCC"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"arc_test.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict) -> Dict:
        choices = item['choices']
        answer = item['answerKey']
        if len(choices['text']) == 3:
            prompt = PROMPT_THREE_OPTIONS.format(
                question=item['question'], 
                A=choices['text'][0], 
                B=choices['text'][1], 
                C=choices['text'][2], 
            )
        elif len(choices['text']) == 5:
            prompt = PROMPT_FIVE_OPTIONS.format(
                question=item['question'], 
                A=choices['text'][0], 
                B=choices['text'][1], 
                C=choices['text'][2], 
                D=choices['text'][3], 
                E=choices['text'][4], 
            )
        elif len(choices['text']) == 4:
            prompt = PROMPT_FOUR_OPTIONS.format(
                question=item['question'], 
                A=choices['text'][0], 
                B=choices['text'][1], 
                C=choices['text'][2], 
                D=choices['text'][3], 
            )
        else:
            raise ValueError(f"Invalid number of choices: {len(choices['text'])}")
        
        return {"prompt": prompt, "answer": answer}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        return self.extract_normal_answer(text=raw_data, answer_pattern=ANSWER_PATTERN)
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['answer']
        
        prediction = self.extract_raw_answer(raw_data=output_text)
        
        is_correct = answer == prediction
        
        return {
            "prediction": prediction,
            "ground_truth": answer,
            "is_correct": is_correct
        }
    
