import os
from collections import Counter
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.deepscaler_rm import (extract_answer, grade_answer_mathd,
                                    grade_answer_sympy)

disable_progress_bars()

DATA_DIR = "data/Mathbench"

CLOZE_PROMPT = """Solve the following math problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.

{question}

Remember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.
""".strip()

SINGLE_CHOICE_PROMPT = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Let's think step by step.
""".strip()

CLOZE_ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

SINGLE_CHOICE_ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?"

class MathBenchEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "MATHBENCH"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        # single choice
        single_choice_data = self.load_jsonl(os.path.join(DATA_DIR, f"college.jsonl"))
        single_choice_data = Dataset.from_list(single_choice_data)
        single_choice_data = single_choice_data.map(lambda x: self.format_prompt(x, type="single_choice"))
        
        # Add origin_query field
        single_choice_data = single_choice_data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
            
        return single_choice_data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict, type: str):
        if type == "cloze":
            # [question: str, answer: str]
            prompt = CLOZE_PROMPT.replace("{question}", item["question"])
        elif type == "single_choice":
            # [question: str, options: list[str]]
            prompt = SINGLE_CHOICE_PROMPT.replace("{question}", item["question"]).replace("{option_a}", item["options"][0]).replace("{option_b}", item["options"][1]).replace("{option_c}", item["options"][2]).replace("{option_d}", item["options"][3])
        else:
            raise ValueError(f"Invalid type: {type}")
        
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str, type: str = "single_choice") -> str:
        if type == "single_choice":
            answer = self.extract_normal_answer(text=raw_data, answer_pattern=SINGLE_CHOICE_ANSWER_PATTERN)
        else:  # cloze type
            if "Final Answer" in raw_data and "\\boxed" not in raw_data:
                answer = self.extract_normal_answer(text=raw_data, answer_pattern=CLOZE_ANSWER_PATTERN)
            else:
                answer = extract_answer(passage=raw_data)
        if answer is None:
            answer = ""
        return answer
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['answer']
        question_type = kwargs.get('question_type', 'single_choice')
        
        prediction = self.extract_raw_answer(raw_data=output_text, type=question_type)
        
        if question_type == "single_choice":
            is_correct = answer == prediction
            ground_truth = answer
        else:  # cloze type
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
    
    
