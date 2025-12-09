import os
import re
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.deepscaler_rm import (extract_answer, grade_answer_mathd,
                                    grade_answer_sympy)

disable_progress_bars()


DATA_DIR = "data/FinQA"

# PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

# {Question}

# Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
# """.strip()

PROMPT = """Solve the following problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.

{Question}

Remember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class FinQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "FinQA"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def list_to_simple_table(self, table_data):
        if not table_data or not isinstance(table_data, list):
            return "Invalid table data"
        
        # Initialize the table string
        table_str = ""
        
        # Process each row
        for row in table_data:
            # Convert each cell in the row to string and strip whitespace
            row_str = [str(cell).strip() for cell in row]
            
            # Join cells with pipe separators and add to table
            table_str += "| " + " | ".join(row_str) + " |\n"
        
        return table_str
    
    def format_prompt(self, item: Dict):
        pre_text = ''.join(item['pre_text'])
        table = item['table']
        table_str = self.list_to_simple_table(table)
        
        post_text = ''.join(item['post_text'])  
        question = item['question']
        Question = f"{pre_text}\n\nHere is the table:\n{table_str}\n\n{post_text}\n\nBased on the content and the table, please answer the following question: {question}\n"
        prompt = PROMPT.replace("{Question}", Question)
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        answer = extract_answer(passage=raw_data)
        if answer is None:
            answer = ""
        return answer
    
    def extract_number(self, text: str):
        pattern = r'-?\d+(?:\.\d+)?'

        match = re.search(pattern, text)
        if match:
            number = float(match.group(0))
            number = round(number, 0)
            return str(number)
        else:
            return text
        
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['answer']
        
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
        
        # finqa has the number of decimal places, so we need to check if the prediction is close to the ground truth
        if not is_correct:
            formatted_prediction = self.extract_number(prediction)
            formatted_ground_truth = self.extract_number(ground_truth)
            if formatted_prediction == formatted_ground_truth:
                is_correct = True
            else:
                is_correct = False
        
        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        }
    
