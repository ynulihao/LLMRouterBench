"""
SGI-Bench Utility Functions

Functions copied/adapted from official SGI-Bench evaluation code to ensure
alignment with official evaluation logic.
"""

import ast
import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from itertools import combinations

from loguru import logger


# ============================================================================
# Answer Parser (from utils.py AnswerPaser class)
# ============================================================================

# Prompt template for answer parsing (official: utils.py lines 319-337)
ANSWER_PARSER_SYSTEM_PROMPT = """You are an expert in structured data parsing. Your task is to convert text content into a standardized structured output based on a provided example data structure.

### Instructions

1.  **Analyze Example Structure:** Carefully analyze the example data structure provided within the `<example>` tags (e.g., it can be a dictionary, list, string, or single character) to understand the desired output format and hierarchy.
2.  **Determine Output Type:** Ensure the overall data type of the final output strictly adheres to the type specified within the `<type>` tags.
3.  **Transform Content:** Parse the text content from the `<input_text>` tags and transform it into a structured output that precisely matches the data format and content defined by `<example>`.
4.  **Preserve Semantics:** During the transformation process, only adjust the format and structure; do not alter the original semantic content of the text within the `<input_text>` tags.
5.  **Ignore Explanatory Text:** If the content within the `<input_text>` tags includes additional explanatory text or descriptions, ignore them and only extract and parse the core, final output data.
6.  **Clean Output:** Your final output must contain only the transformed structured content, without any additional explanations, descriptions, or irrelevant text and symbols.

<example>
{example}
</example>

<type>
{type_hint}
</type>"""


class AnswerParser:
    """
    Answer parser for normalizing model outputs to match expected format.

    Per official implementation (utils.py:314-357), uses an LLM to convert
    model outputs to standardized format matching the expected answer structure.

    This is used in Deep Research task to normalize answers before exact match
    comparison (official: step_2_score.py line 77 uses model_answer_after_llm_paser).
    """

    def __init__(self, generator=None):
        """
        Initialize AnswerParser.

        Args:
            generator: LLM generator to use for parsing. If None, will attempt
                      to create one using GRADER_* environment variables.
        """
        self.generator = generator
        self._initialized = generator is not None

    def _ensure_generator(self):
        """Lazy initialization of generator if not provided."""
        if self._initialized:
            return True

        try:
            from generators.generator import DirectGenerator

            model_name = os.getenv("GRADER_MODEL", "gpt-4.1-mini")
            base_url = os.getenv("GRADER_BASE_URL", "")
            api_key = os.getenv("GRADER_API_KEY", "")

            if not base_url or not api_key:
                logger.warning("AnswerParser: GRADER_BASE_URL or GRADER_API_KEY not set")
                return False

            self.generator = DirectGenerator(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0.0,
                timeout=60
            )
            self._initialized = True
            return True

        except Exception as e:
            logger.warning(f"AnswerParser: Failed to initialize generator: {e}")
            return False

    def _get_type_hint(self, example: Union[str, list, dict]) -> str:
        """Get type hint string for the example."""
        if isinstance(example, str) and len(example) == 1:
            return "One letter"
        return str(type(example).__name__)

    def _format_example(self, example: Union[str, list, dict]) -> str:
        """Format example for prompt."""
        if isinstance(example, (list, dict)):
            return json.dumps(example, indent=4)
        return str(example)

    def parse(self, text: str, example: Union[str, list, dict]) -> Optional[str]:
        """
        Parse and normalize text to match example format.

        Args:
            text: Raw text to parse (model output)
            example: Example of expected output format

        Returns:
            Normalized output string, or None if parsing fails
        """
        if not self._ensure_generator():
            return None

        if not isinstance(text, str):
            text = str(text)

        # Extract from <answer> tags first (official: line 350)
        final_answer = extract_final_answer(text)
        if final_answer is None:
            final_answer = text

        # Build prompt (official: lines 318-344)
        system_prompt = ANSWER_PARSER_SYSTEM_PROMPT.format(
            example=self._format_example(example),
            type_hint=self._get_type_hint(example)
        )

        query = f"""
<input_text>
{final_answer}
</input_text>
"""

        try:
            result = self.generator.generate(system_prompt + "\n" + query)
            return result.output.strip()
        except Exception as e:
            logger.warning(f"AnswerParser: Generation failed: {e}")
            return None

    def __call__(self, text: str, example: Union[str, list, dict]) -> Optional[str]:
        """Callable interface for compatibility."""
        return self.parse(text, example)


# ============================================================================
# Common Utilities (from utils.py)
# ============================================================================

def extract_final_answer(answer_with_thinking: str, start_tag: str = '<answer>', end_tag: str = '</answer>') -> Optional[str]:
    """
    Extract content between answer tags.
    Uses rfind to get the LAST occurrence of the start tag.
    """
    if answer_with_thinking is None:
        return None
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[start_index + len(start_tag):end_index].strip()
    return None


def check_syntax(code_string: str) -> bool:
    """Check if Python code has valid syntax."""
    try:
        compile(code_string, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def get_function_lines(file_content: str) -> Dict[str, Tuple[int, int]]:
    """Get line ranges for all functions in code."""
    node = ast.parse(file_content)
    function_lines = {}

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            func_name = item.name
            start_line = item.lineno
            end_line = item.end_lineno
            function_lines[func_name] = (start_line, end_line)

    return function_lines


def replace_code(content_1: str, start_line_1: int, end_line_1: int,
                 content_2: str, start_line_2: int, end_line_2: int) -> str:
    """Replace code lines from one content with another."""
    lines_1 = content_1.splitlines(keepends=True)
    lines_2 = content_2.splitlines(keepends=True)
    lines_1[start_line_1 - 1:end_line_1] = lines_2[start_line_2 - 1:end_line_2]
    return ''.join(lines_1)


def replace_function(main_code: str, new_code: str, function_name: str) -> str:
    """
    Replace a function in main_code with the same function from new_code.

    Args:
        main_code: Original code containing the function to replace
        new_code: Code containing the new function implementation
        function_name: Name of the function to replace

    Returns:
        Updated main_code with replaced function

    Raises:
        AssertionError: If code syntax is invalid
    """
    assert check_syntax(main_code), "wrong main_code"
    assert check_syntax(new_code), "wrong new_code"

    functions_dict_1 = get_function_lines(main_code)
    functions_dict_2 = get_function_lines(new_code)

    start_line_1, end_line_1 = functions_dict_1[function_name]
    start_line_2, end_line_2 = functions_dict_2[function_name]

    main_code_after_replacing = replace_code(
        main_code, start_line_1, end_line_1,
        new_code, start_line_2, end_line_2
    )
    assert check_syntax(main_code_after_replacing), "wrong main_code after replacing"
    return main_code_after_replacing


# ============================================================================
# Idea Generation Utilities (from task_2 step_2_score.py)
# ============================================================================

def format_idea_data(idea_data: Dict) -> str:
    """Format idea dictionary to text representation."""
    fields = [
        "Idea",
        "ImplementationSteps",
        "ImplementationOrder",
        "Dataset",
        "EvaluationMetrics",
        "ExpectedOutcome"
    ]

    formatted_text = ""
    for field in fields:
        if field in idea_data and idea_data[field]:
            formatted_text += f"{field}: {idea_data[field]}\n\n"

    return formatted_text.strip()


def get_context_from_data(data: Dict) -> str:
    """Extract context fields from data for evaluation prompt."""
    context_fields = [
        "related_work",
        "challenge",
        "limitation",
        "motivation",
        "task_objective",
        "existing_solutions"
    ]
    context = ""
    for field in context_fields:
        if field in data and data[field]:
            context += f"{field}: {data[field]}\n\n"

    return context.strip()


def flip_evaluation_result(result: Dict) -> Dict:
    """Flip evaluation result when positions are swapped."""
    flipped = {}
    mapping = {
        "win_A": "win_B",
        "win_B": "win_A"
    }

    for key, value in result.items():
        if isinstance(value, dict) and "judgment" in value:
            flipped[key] = {
                "judgment": mapping.get(value["judgment"], value["judgment"]),
                "reason": value.get("reason", "")
            }
        else:
            flipped[key] = mapping.get(value, value)

    return flipped


def extract_win_lose(result_text: str, dimension: str) -> Optional[str]:
    """Extract win/lose judgment from evaluation result text."""
    pattern = rf"{dimension}\s*:\s*\[\s*(Win\s*A|Win\s*B)\s*\]"
    match = re.search(pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    backup_pattern = rf"{dimension}\s*:\s*(Win\s*A|Win\s*B)\s+"
    match = re.search(backup_pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    line_pattern = rf"{dimension}[^\n]*?(Win\s*A|Win\s*B)"
    match = re.search(line_pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    return None


def extract_reason(result_text: str, dimension: str) -> str:
    """Extract reason from evaluation result text."""
    pattern = rf"{dimension}\s*:\s*\[[^\]]+\]\s*because\s*(.*?)(?=\n\w|$)"
    match = re.search(pattern, result_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    backup_pattern = rf"{dimension}\s*:[^\n]*?(because|due to|as|since)([^\n]+)"
    match = re.search(backup_pattern, result_text, re.IGNORECASE)
    if match:
        return match.group(2).strip()

    fallback_pattern = rf"{dimension}\s*:[^\n]*(.*?)(?=\n\w+:|$)"
    match = re.search(fallback_pattern, result_text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        reason = re.sub(r"\[(Win\s*A|Win\s*B)\]", "", text).strip()
        return reason

    return "No specific reason provided"


def parse_evaluation_result(result: str) -> Optional[Dict]:
    """Parse evaluation result from LLM judge output."""
    dimensions = ["effectiveness", "novelty", "detailedness", "feasibility", "overall"]
    parsed_results = {}
    all_valid = True

    for dim in dimensions:
        judgment = extract_win_lose(result, dim.capitalize())
        reason = extract_reason(result, dim.capitalize())

        if judgment is None:
            all_valid = False
            break

        parsed_results[dim] = {
            "judgment": judgment,
            "reason": reason
        }

    if not all_valid:
        return None

    return parsed_results


# ============================================================================
# Wet Experiment Utilities (from task_3_wet_experiment step_2_score.py)
# ============================================================================

def parse_experiment_steps(text: str) -> List[Dict]:
    """
    Parse experiment protocol steps from text.

    Format: variable_name = <action_name>(parameter_list)

    Args:
        text: Protocol text to parse

    Returns:
        List of step dictionaries with 'action', 'input', 'output' keys
    """
    # Match format: variable_name = <action_name>(parameter_list)
    step_pattern = r'(\w+)\s*=\s*<([^>]+)>\(\s*([\s\S]*?)(?=\n\s*\)\s*$)'
    # Match format: key=value or key=value,
    param_pattern = r'^\s*(\w+)\s*=\s*(.*?)\s*(?:,)?\s*$'

    steps = []

    for match in re.finditer(step_pattern, text, re.MULTILINE):
        output_var = match.group(1).strip()
        action_name = match.group(2).strip()
        params = match.group(3).strip()

        param_dict = {}
        param_lines = [line.strip() for line in params.split('\n')
                      if line.strip() and line.strip() != ')']

        for line in param_lines:
            param_match = re.match(param_pattern, line)
            if param_match:
                key = param_match.group(1)
                value = param_match.group(2).strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                param_dict[key] = value

        steps.append({
            "action": action_name,
            "input": param_dict,
            "output": output_var
        })

    return steps


def identify_variable_types(steps: List[Dict]) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    """
    Identify raw and generated variables in experiment steps.

    Returns:
        original_vars: Set of raw variables (never appear as outputs)
        generated_vars: Set of generated variables (appear as outputs)
        output_to_step_map: Mapping from output variable to step index
    """
    generated_vars = set()
    all_input_vars = set()
    output_to_step_map = {}

    for idx, step in enumerate(steps):
        output_var = step["output"]
        generated_vars.add(output_var)
        output_to_step_map[output_var] = idx

        for input_val in step["input"].values():
            # Check if it's a variable (not a string literal or number)
            if isinstance(input_val, str) and \
               not (input_val.startswith('"') and input_val.endswith('"')) and \
               not (input_val.replace('.', '', 1).isdigit() or
                    (input_val.startswith('-') and input_val[1:].replace('.', '', 1).isdigit())):
                all_input_vars.add(input_val)

    original_vars = all_input_vars - generated_vars
    return original_vars, generated_vars, output_to_step_map


def kendall_tau_distance(seq1: List, seq2: List) -> float:
    """
    Calculate Kendall tau distance (normalized similarity) between two sequences.

    Returns:
        Similarity score between 0 and 1 (1 = identical ordering)
    """
    if len(seq1) != len(seq2):
        return 0.0
    n = len(seq1)
    if n <= 1:
        return 1.0

    inversions = 0
    for i, j in combinations(range(n), 2):
        if (seq1[i] < seq1[j] and seq2[i] > seq2[j]) or \
           (seq1[i] > seq1[j] and seq2[i] < seq2[j]):
            inversions += 1

    max_inversions = n * (n - 1) / 2
    return 1.0 - (inversions / max_inversions if max_inversions > 0 else 0.0)


def compare_exp_steps(gt_steps: List[Dict], pred_steps: List[Dict]) -> Dict:
    """
    Compare ground truth steps with predicted steps.

    Args:
        gt_steps: Ground truth experiment steps
        pred_steps: Predicted experiment steps

    Returns:
        Dictionary with:
        - order_similarity: Kendall tau distance of action sequences
        - parameter_acc: Parameter accuracy (1 - error_rate)
        - details: List of per-step comparison details
    """
    results = {
        "order_similarity": 0.0,
        "parameter_acc": 0.0,
        "details": []
    }

    actions_gt = [step["action"] for step in gt_steps]
    actions_pred = [step["action"] for step in pred_steps]

    results["order_similarity"] = kendall_tau_distance(actions_gt, actions_pred)

    # Identify variable types
    original_vars_gt, generated_vars_gt, output_to_step_map_gt = identify_variable_types(gt_steps)
    original_vars_pred, generated_vars_pred, output_to_step_map_pred = identify_variable_types(pred_steps)

    # Variable mapping from pred to gt
    var_map_pred2gt = {}

    error_count = 0
    min_len = min(len(gt_steps), len(pred_steps))

    for i in range(min_len):
        step_gt = gt_steps[i]
        step_pred = pred_steps[i]
        detail = {
            "step": i + 1,
            "action_gt": step_gt["action"],
            "action_pred": step_pred["action"],
            "status": "success",
            "message": ""
        }

        # Check action names
        if step_gt["action"] != step_pred["action"]:
            detail["status"] = "error"
            detail["message"] += f"Action mismatch: expected '{step_gt['action']}', got '{step_pred['action']}'. "
            error_count += 1
            results["details"].append(detail)
            continue

        # Check parameter keys
        keys_gt = set(step_gt["input"].keys())
        keys_pred = set(step_pred["input"].keys())
        if keys_gt != keys_pred:
            detail["status"] = "error"
            detail["message"] += f"Parameter keys mismatch: expected {keys_gt}, got {keys_pred}. "
            error_count += 1
            results["details"].append(detail)
            continue

        # Check parameter values
        is_step_error = False
        for key in keys_gt:
            value_gt = step_gt["input"][key]
            value_pred = step_pred["input"][key]

            is_input_var_gt_generated = value_gt in generated_vars_gt
            is_input_var_pred_generated = value_pred in generated_vars_pred

            # Case 1: Both are generated variables
            if is_input_var_gt_generated and is_input_var_pred_generated:
                mapped_value_pred = var_map_pred2gt.get(value_pred)
                if mapped_value_pred != value_gt:
                    detail["status"] = "error"
                    detail["message"] += f"Parameter '{key}' generated variable reference mismatch. "
                    is_step_error = True
            # Case 2: Both are raw variables (allow different values)
            elif not is_input_var_gt_generated and not is_input_var_pred_generated:
                pass
            # Case 3: Type mismatch
            else:
                detail["status"] = "error"
                detail["message"] += f"Parameter '{key}' type mismatch. "
                is_step_error = True

        if not is_step_error:
            var_map_pred2gt[step_pred["output"]] = step_gt["output"]
        else:
            error_count += 1

        results["details"].append(detail)

    # Handle length mismatch
    if len(gt_steps) != len(pred_steps):
        error_count += abs(len(gt_steps) - len(pred_steps))
        if len(pred_steps) > len(gt_steps):
            for i in range(min_len, len(pred_steps)):
                results["details"].append({
                    "step": i + 1,
                    "action_gt": None,
                    "action_pred": pred_steps[i]["action"],
                    "status": "error",
                    "message": "Extra step."
                })
        elif len(gt_steps) > len(pred_steps):
            for i in range(min_len, len(gt_steps)):
                results["details"].append({
                    "step": i + 1,
                    "action_gt": gt_steps[i]["action"],
                    "action_pred": None,
                    "status": "error",
                    "message": "Missing step."
                })

    max_steps = max(len(gt_steps), len(pred_steps))
    results["parameter_acc"] = 1 - (error_count / max_steps) if max_steps > 0 else 1.0

    return results


# ============================================================================
# JSON Parsing Utilities (for Idea Generation)
# ============================================================================

def parse_generated_idea(text: str) -> Optional[Dict]:
    """
    Parse generated idea from model output.
    Tries JSON parsing first, then regex fallback.

    Args:
        text: Model output text

    Returns:
        Parsed idea dictionary or None if parsing fails
    """
    # Try to extract from <answer> tags first
    answer_content = extract_final_answer(text)
    if answer_content:
        text = answer_content

    # Try JSON code block extraction
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try direct JSON parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON-like content
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            return json.loads(text[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass

    # Regex fallback for each field
    result = {}

    # Extract Idea
    idea_match = re.search(r'"?Idea"?\s*:\s*"([^"]*)"', text, re.IGNORECASE)
    if idea_match:
        result["Idea"] = idea_match.group(1)

    # Extract other fields similarly...
    fields = ["ImplementationSteps", "ImplementationOrder", "Dataset",
              "EvaluationMetrics", "ExpectedOutcome"]

    for field in fields:
        pattern = rf'"{field}"\s*:\s*(\{{[^}}]*\}}|\[[^\]]*\]|"[^"]*")'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                result[field] = json.loads(match.group(1))
            except json.JSONDecodeError:
                result[field] = match.group(1).strip('"')

    return result if result else None


# ============================================================================
# Graph Similarity Utilities (for Idea Generation - Feasibility Objective)
# ============================================================================

def edge_jaccard(G1, G2) -> float:
    """
    Calculate Jaccard similarity of edges between two graphs.

    Args:
        G1: First networkx DiGraph
        G2: Second networkx DiGraph

    Returns:
        Jaccard similarity score (0-1)
    """
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    if not edges1 and not edges2:
        return 1.0
    if not edges1 or not edges2:
        return 0.0
    return len(edges1 & edges2) / len(edges1 | edges2)


def node_text_similarity(G1, G2) -> float:
    """
    Calculate text similarity between nodes of two graphs using word overlap.

    Args:
        G1: First networkx DiGraph with 'text' attribute on nodes
        G2: Second networkx DiGraph with 'text' attribute on nodes

    Returns:
        Word overlap similarity score (0-1)
    """
    texts1 = [G1.nodes[n].get('text', '') for n in G1.nodes()]
    texts2 = [G2.nodes[n].get('text', '') for n in G2.nodes()]

    if not texts1 or not texts2:
        return 0.0

    combined_text1 = ' '.join(str(t) for t in texts1)
    combined_text2 = ' '.join(str(t) for t in texts2)

    if len(combined_text1.strip()) < 3 or len(combined_text2.strip()) < 3:
        return 0.0

    words1 = set(combined_text1.lower().split())
    words2 = set(combined_text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def graph_similarity(dict1: Dict, dict2: Dict, alpha: float = 0.5) -> float:
    """
    Calculate graph similarity between two idea implementations.

    Builds directed graphs from ImplementationSteps (nodes) and
    ImplementationOrder (edges), then computes:
    - Edge Jaccard similarity (structural similarity)
    - Node text similarity (content similarity)

    Final score = alpha * edge_sim + (1 - alpha) * text_sim

    Args:
        dict1: First idea dict with 'ImplementationSteps' and 'ImplementationOrder'
        dict2: Second idea dict with 'ImplementationSteps' and 'ImplementationOrder'
        alpha: Weight for edge similarity (default 0.5)

    Returns:
        Combined similarity score (0-1)
    """
    try:
        import networkx as nx
    except ImportError:
        # networkx not available, return default score
        return 0.5

    # Check required keys
    required_keys = ["ImplementationSteps", "ImplementationOrder"]
    if not all(k in dict1 for k in required_keys) or \
       not all(k in dict2 for k in required_keys):
        return 0.0

    if not dict1["ImplementationSteps"] or not dict1["ImplementationOrder"] or \
       not dict2["ImplementationSteps"] or not dict2["ImplementationOrder"]:
        return 0.0

    try:
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()

        # Add nodes from ImplementationSteps
        steps1 = dict1["ImplementationSteps"]
        steps2 = dict2["ImplementationSteps"]

        if isinstance(steps1, dict):
            for k, v in steps1.items():
                G1.add_node(str(k), text=str(v))
        elif isinstance(steps1, list):
            for i, v in enumerate(steps1):
                G1.add_node(str(i), text=str(v))

        if isinstance(steps2, dict):
            for k, v in steps2.items():
                G2.add_node(str(k), text=str(v))
        elif isinstance(steps2, list):
            for i, v in enumerate(steps2):
                G2.add_node(str(i), text=str(v))

        if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
            return 0.0

        def process_order_items(order_list, graph, step_keys):
            """Process ImplementationOrder to add edges to graph."""
            edges_added = False
            order_list = [str(o) for o in order_list] if isinstance(order_list, list) else []

            if all(o.isdigit() for o in order_list):
                # Sequential order: [1, 2, 3] -> 1->2->3
                nodes = sorted([o for o in order_list if o in step_keys])
                for i in range(len(nodes) - 1):
                    graph.add_edge(nodes[i], nodes[i+1])
                    edges_added = True
            else:
                # Explicit edges: ["1-2", "2-3"]
                for o in order_list:
                    if "-" in str(o):
                        try:
                            parts = str(o).split("-")
                            if len(parts) == 2:
                                src, dst = parts
                                if src in step_keys and dst in step_keys:
                                    graph.add_edge(src, dst)
                                    edges_added = True
                        except Exception:
                            pass
            return edges_added

        step_keys_1 = set(str(k) for k in (steps1.keys() if isinstance(steps1, dict) else range(len(steps1))))
        step_keys_2 = set(str(k) for k in (steps2.keys() if isinstance(steps2, dict) else range(len(steps2))))

        edges_added_G1 = process_order_items(dict1["ImplementationOrder"], G1, step_keys_1)
        edges_added_G2 = process_order_items(dict2["ImplementationOrder"], G2, step_keys_2)

        # Fallback: if no edges from ImplementationOrder, build sequential edges
        # Official implementation (lines 140-150): auto-construct sequential edges
        if not edges_added_G1:
            nodes1 = sorted([n for n in G1.nodes()])
            for i in range(len(nodes1) - 1):
                G1.add_edge(nodes1[i], nodes1[i + 1])
                edges_added_G1 = True

        if not edges_added_G2:
            nodes2 = sorted([n for n in G2.nodes()])
            for i in range(len(nodes2) - 1):
                G2.add_edge(nodes2[i], nodes2[i + 1])
                edges_added_G2 = True

        # If still no edges (single node graphs), only compute text similarity
        if not edges_added_G1 or not edges_added_G2:
            return node_text_similarity(G1, G2)

        edge_sim = edge_jaccard(G1, G2)
        text_sim = node_text_similarity(G1, G2)

        return alpha * edge_sim + (1 - alpha) * text_sim

    except Exception:
        return 0.0
