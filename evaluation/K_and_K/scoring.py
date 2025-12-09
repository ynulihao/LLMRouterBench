import re
from collections import defaultdict

import numpy as np
from loguru import logger
from collections import Counter
from evaluation.K_and_K.prompt import (demonstration_2char,
                                     demonstration_2char_no_reason,
                                     system_instruction,
                                     system_instruction_no_reason)


def ensemble_answers(answer_list) -> tuple:
    """集成多个答案，返回最终答案和所有答案的一致率
    
    Args:
        answer_list: 答案列表
    Returns:
        final_answer: 格式化后的最终答案
        agreement_rates: 字典，每个答案及其一致率
    """
    # 1. 收集所有有效的完整答案
    valid_solutions = []
    for answer_str in answer_list:
        try:
            solution = parse_single_answer(answer_str)
            if solution:  # 确保解析出有效答案
                valid_solutions.append(solution)
        except Exception as e:
            logger.error(f"Error parsing answer: {e}")
            continue
    
    if not valid_solutions:
        return "", {"": {"frequency": 1, "count": 1}}
    
    # 2. 将解决方案标准化为字符串以便比较
    solution_strings = []
    solution_map = {}
    for solution in valid_solutions:
        sorted_items = sorted(solution.items())
        solution_str = ';'.join(f"{name}:{role}" for name, role in sorted_items)
        solution_strings.append(solution_str)
        solution_map[solution_str] = solution
    
    # 3. 统计和分析
    solution_counts = Counter(solution_strings)
    total_solutions = len(valid_solutions)
    
    # 4. 计算每个答案的一致率
    agreement_rates = {}
    for solution_str, count in solution_counts.items():
        agreement_rate = count / total_solutions
        # 使用原始解决方案作为键
        solution = solution_map[solution_str]
        # 将解决方案转换为更易读的格式作为键
        formatted_solution = format_final_answer(solution)
        agreement_rates[formatted_solution] = dict(frequency=agreement_rate, count=count)
    
    # 5. 获取最高一致率的解决方案
    max_count = max(solution_counts.values())
    top_solutions = [
        solution_str for solution_str, count in solution_counts.items()
        if count == max_count
    ]
    
    # 6. 选择最终答案并格式化
    final_solution = solution_map[top_solutions[0]]
    final_answer = format_final_answer(final_solution)
    
    return final_answer, agreement_rates

    
def format_final_answer(result_dict):
    """将结果格式化为原始的格式"""
    if not result_dict:
        return None
    
    names = sorted(result_dict.keys())
    formatted_lines = []
    for i, name in enumerate(names, 1):
        role = result_dict[name]
        role_cap = role[0].upper() + role[1:]
        formatted_lines.append(f"({i}) {name} is a {role_cap}.")
    
    return "\n".join(formatted_lines)

def parse_single_answer(answer_str) -> dict:
    """解析单个答案字符串，返回人物及其身份的字典，支持多种格式"""
    result = {}
    answer_str, is_success = parse_answer(pred_str=answer_str)
    
    # 1. preprocess: remote the markdown format
    answer_str = re.sub(r'\*\*(.*?)\*\*', r'\1', answer_str)
    # 2. preprocess: remote the comma separated answer
    answer_str = re.sub(r'(?<=[\w\.\}])(\s*),\s*(?=\(\d+\))', '\n', answer_str)
    # 3. preprocess: remote the LaTeX text{} format
    answer_str = re.sub(r'\\text\{(.*?)\}', r'\1', answer_str)
    
    # 4. split each line
    lines = answer_str.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
        
        # 5.1 match (1) Name: Knight format
        match = re.search(r'\((\d+)\)\s*(.*?):\s*(knight|knave)', line, re.IGNORECASE)
        if match:
            number, name, role = match.groups()
            result[name.strip()] = role.lower()
            continue
        
        # 5.2 match origin `is a knight` format
        match = re.search(r'\((\d+)\)\s*(.*?)\s+is\s+a\s+(knight|knave)\.?', line, re.IGNORECASE)
        if match:
            number, name, role = match.groups()
            result[name.strip()] = role.lower()
            continue
        # 5.3 try more relaxed match, not depend on the number
        match = re.search(r'(.*?)\s+is\s+a\s+(knight|knave)\.?', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            result[name.strip()] = role.lower()
    
    return result

def judge_answer(pred_answer, reformat_gold_conditions, finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"]):
    """
    判断提取的答案是否满足给定的标准答案条件。

    参数：
        pred_answer (str): 提取的答案。
        reformat_gold_conditions (list[str]): 标准答案条件列表。
        finish_patterns (list[str]): 用于截断答案的关键词。

    返回：
        is_correct (bool): 答案是否完全正确。
        wrong_reason (str): 若答案错误，提供错误原因。
        correct_ratio (float): 答案中匹配标准答案条件的比例。
    """
    correct_count = 0
    wrong_reason = ""

    beyond_id = len(reformat_gold_conditions) + 1
    beyond_id_pattern = f"({beyond_id})"

    for finish_pattern in finish_patterns:
        if finish_pattern in pred_answer:
            pred_answer = pred_answer.split(finish_pattern)[0]

    if beyond_id_pattern in pred_answer:
        return False, "beyond_list", 0.0

    if "if" in pred_answer:
        return False, "contain_if", 0.0

    for gold_condition in reformat_gold_conditions:
        if gold_condition.lower() in pred_answer.lower():
            correct_count += 1

    correct_ratio = correct_count / len(reformat_gold_conditions)

    if correct_count == len(reformat_gold_conditions):
        return True, "", correct_ratio
    else:
        return False, "wrong_identity", correct_ratio
    
def parse_answer(
    pred_str, 
    conclusion_patterns=['CONCLUSION:', 'boxed', 'Conclusion:', 'conclusion:'], 
    finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"]
):
    """
    从模型的生成文本中提取结论部分（即模型给出的最终答案）。

    参数：
        pred_str (str): 模型生成的文本。
        conclusion_patterns (list[str]): 用于定位结论的关键词。
        finish_patterns (list[str]): 可能用于截断答案的关键词。

    返回：
        pred_answer (str): 提取后的结论文本。
        matched_conclusion (bool): 是否成功匹配到结论关键词。
    """
    pred_str = pred_str.split("### Question")[0]

    for pattern in conclusion_patterns:
        parts = pred_str.split(pattern)
        if len(parts) > 1 and len(parts[1].strip()) > 0:
            pred_answer = parts[1]
            matched_conclusion = True
            break
    else:
        pred_answer = pred_str
        matched_conclusion = False

    # 根据finish_patterns截断答案
    for finish_pattern in finish_patterns:
        if finish_pattern in pred_answer:
            pred_answer = pred_answer.split(finish_pattern)[0]

    return pred_answer.strip(), matched_conclusion

def parse_cot_eval(pred_str, ans,
                   conclusion_patterns=['CONCLUSION:', 'boxed', 'Conclusion:', 'conclusion:'],
                   verbose=False,
                   finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"],
                   reformat_gold_conditions=None):
    
    def judge_string(input_str, reformat_gold_conditions, wrong_reason, finish_patterns):
        correct_count = 0
        is_correct = False
        beyond_id = len(reformat_gold_conditions)+1
        beyond_id_pattern = f"({beyond_id})"

        for finish_pattern in finish_patterns:
            if finish_pattern in input_str:
                input_str = input_str.split(finish_pattern)[0]

        if beyond_id_pattern in input_str:
            is_correct = False
            wrong_reason = "beyond_list"
        elif "if" in input_str:
            is_correct = False
            wrong_reason = "contain_if"
        else:
            is_correct = True
            for gold_condition in reformat_gold_conditions:
                if gold_condition not in input_str:
                    is_correct = False
                    wrong_reason = "wrong_identity"
                else:
                    correct_count += 1
        correct_ratio = correct_count / len(reformat_gold_conditions)

        return is_correct, wrong_reason, correct_ratio

    def check_numbers_in_string(s, N):
        for i in range(1, N + 1):
            if f"({i})" not in s:
                return False
        return True
    
    original_str = pred_str
    pred_str = pred_str.split("### Question")[0]
    pred_answer = pred_str
    is_correct = False
    correct_ratio = 0
    if reformat_gold_conditions is None:
        gold = ans.replace(" and ", "").replace(".", "")
        gold_conditions = gold.split(",")
        reformat_gold_conditions = []
        for condition in gold_conditions:
            gold_condition = condition.strip()    # Remove leading and trailing spaces
            reformat_gold_conditions.append(gold_condition)

    wrong_reason = "no_conclusion_matched"
    for pattern in conclusion_patterns:
        pred = pred_str.split(pattern)
        if len(pred) > 1:
            if len(pred[1]) > 0:  # if the matched the answer is not empty
                pred_answer = pred[1]
                is_correct, wrong_reason, correct_ratio = judge_string(
                    pred_answer, reformat_gold_conditions, wrong_reason, finish_patterns)
                break
    if is_correct == False and wrong_reason == "no_conclusion_matched": 
        if check_numbers_in_string(pred_str, len(reformat_gold_conditions)): # the answer contains (1)..(2)..
            is_correct, wrong_reason, correct_ratio = judge_string(
                pred_str, reformat_gold_conditions, wrong_reason, finish_patterns)
    if is_correct == False and verbose == True:
        print("wrong_reason:",wrong_reason)
        print("********* \nprediction before parse:\n", original_str)
        print("********* \nprediction after parse:\n", pred_answer)

    return is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions


class KKProcessor:
    def __init__(self, cot=True, no_linebreak=True):
        self.cot = cot
        self.no_linebreak = no_linebreak

    def format_example(self, test_records, idx, model_name=None):
       
        item = test_records[idx]

        prompt = "### Question: "+item["quiz"] + "\n"
        if self.cot:
            if model_name in ["deepseek-ai/deepseek-math-7b-instruct", "AI-MO/NuminaMath-7B-CoT"]:
                prompt += "Please reason step by step, and put your final answer within \\boxed{}."
            else:
                prompt += "### Answer: Let's think step by step"
        else:
            if self.no_linebreak:
                prompt += "### Answer:"
            else:
                prompt += "### Answer:\n"
        answer = item["solution_text"]
        return prompt, answer

    def gen_test_prompt(self, ntrain, test_records, idx, model_name=None):
        if self.cot:
            train_prompt = system_instruction
        else:
            train_prompt = system_instruction_no_reason

        if ntrain == 1:
            if self.cot:
                train_prompt += "\n\n"+demonstration_2char
            else:
                train_prompt += "\n\n"+demonstration_2char_no_reason
        elif ntrain > 1:
            raise NotImplementedError

        prompt_end, answer = self.format_example(test_records, idx, model_name)
        prompt = train_prompt + "\n\n" + prompt_end

        return prompt, answer

    def _parse_cot_eval(self, pred_str, ans, model_name=None):
        conclusion_patterns = ['CONCLUSION:', 'Conclusion:', 'conclusion:']

        if model_name in ["deepseek-ai/deepseek-math-7b-instruct", "AI-MO/NuminaMath-7B-CoT"]:
            conclusion_patterns = ['boxed{', 'CONCLUSION:', 'Conclusion:', 'conclusion:']

        is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions = parse_cot_eval(
            pred_str, ans, conclusion_patterns=conclusion_patterns, verbose=False)

        return is_correct, pred_answer, reformat_gold_conditions