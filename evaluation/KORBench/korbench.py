import os
import json
from typing import Dict

from datasets import Dataset, disable_progress_bars, concatenate_datasets

import yaml
from evaluation.base_evaluator import BaseEvaluator
from evaluation.KORBench.eval_utils import evaluate_response_vs_answer, extract_single_answer

disable_progress_bars()

DATA_DIR = "data/KORBench"

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError("No JSON or JSONL file found.")
    
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]
    
    if mapping_key:
        return {item[mapping_key]: item for item in data if mapping_key in item}
    else:
        return data
    
class KORBenchEvaluator(BaseEvaluator):
    def __init__(self, split: str = "full"):
        super().__init__()
        self.task = f"KORBench-{split}"
        self.seed = 42
        self.split = split
    
    def load_yaml(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, split: str = "test"):
        task_list = ['cipher', 'operation', 'puzzle', 'counterfactual', 'logic']
        dataset = []
        for task in task_list:
            dataset.append(self.load_single_data(task=task))
        data = concatenate_datasets(dataset)
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def load_single_data(self, task: str):
        if task in ['cipher', 'operation', 'puzzle', 'counterfactual', 'logic']:
            sample = self.load_jsonl(os.path.join(DATA_DIR, task, "sample.jsonl"))
            
            few_shot = read_json_or_jsonl(data_path=os.path.join(DATA_DIR, task), split='three-shot')
            rule = read_json_or_jsonl(data_path=os.path.join(DATA_DIR, task), split="rule", mapping_key="idx")
        else:
            raise ValueError(f"Invalid task: {task}")
        
        template = self.load_yaml(os.path.join(DATA_DIR, "three-shot.yaml"))
        
        data = Dataset.from_list(sample)
        
        data = data.map(
            lambda x: self.format_prompt(item=x, task=task, template=template, rule=rule, few_shot=few_shot)
        )
        
        return data
    
    def format_prompt(self, item, task, template, rule, few_shot):
        rule_id = item['rule_id']
        rule_content = rule[rule_id]['rule_content']
        question = item['question']
        
        few_shot_qa = [
            i for fs in few_shot if fs['rule_id'] == rule_id for i in [fs['question'], fs['answer']]
        ]
        prompt_format = [rule_content, *few_shot_qa, question]
        prompt = template[f'{task}_prompt_format'][0].format(*prompt_format)
        
        return {"prompt": prompt, "question_type": task}
    
    def extract_raw_answer(self, raw_data: str, question_type: str, rule_id: str, idx: str) -> str:
        return extract_single_answer(response=raw_data, question_type=question_type, rule_id=rule_id, idx=idx)
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['answer']
        question_type = data['question_type']
        rule_id = data['rule_id']
        idx = data['idx']
        
        prediction = self.extract_raw_answer(raw_data=output_text, question_type=question_type, rule_id=rule_id, idx=idx)
        
        is_correct = evaluate_response_vs_answer(response=prediction, answer=answer, question_type=question_type, rule_id=rule_id, idx=idx)
        
        return {
            "prediction": prediction,
            "ground_truth": answer,
            "is_correct": is_correct
        }
    