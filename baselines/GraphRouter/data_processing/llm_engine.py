import json
import re
import time
from transformers import GPT2Tokenizer
from utils import model_prompting, f1_score, exact_match_score, get_bert_score
from beartype.typing import Any, Dict, List, Tuple, Optional

# Initialize tokenizer for token counting (used in cost calculation)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class LLMEngine:
    """
    A class to manage interactions with multiple language models and evaluate their performance.

    Handles model selection, querying, cost calculation, and performance evaluation
    using various metrics for different tasks.
    """

    def __init__(self, llm_names: List[str], llm_description: Dict[str, Dict[str, Any]]):
        """
        Initialize the LLM Engine with available models and their descriptions.

        Args:
            llm_names: List of language model names available in the engine
            llm_description: Dictionary containing model configurations and pricing details
                Structure: {
                    "model_name": {
                        "model": "api_identifier",
                        "input_price": cost_per_input_token,
                        "output_price": cost_per_output_token,
                        ...
                    },
                    ...
                }
        """
        self.llm_names = llm_names
        self.llm_description = llm_description

    def compute_cost(self, llm_idx: int, input_text: str, output_size: int) -> float:
        """
        Calculate the cost of a model query based on input and output token counts.

        Args:
            llm_idx: Index of the model in the llm_names list
            input_text: The input prompt sent to the model
            output_size: Number of tokens in the model's response

        Returns:
            float: The calculated cost in currency units
        """
        # Count input tokens
        input_size = len(tokenizer(input_text)['input_ids'])

        # Get pricing information for the selected model
        llm_name = self.llm_names[llm_idx]
        input_price = self.llm_description[llm_name]["input_price"]
        output_price = self.llm_description[llm_name]["output_price"]

        # Calculate total cost
        cost = input_size * input_price + output_size * output_price
        return cost

    def get_llm_response(self, query: str, llm_idx: int) -> str:
        """
        Send a query to a language model and get its response.

        Args:
            query: The prompt text to send to the model
            llm_idx: Index of the model in the llm_names list

        Returns:
            str: The model's text response

        Note:
            Includes a retry mechanism with a 2-second delay if the first attempt fails
        """
        llm_name = self.llm_names[llm_idx]
        model = self.llm_description[llm_name]["model"]

        try:
            response = model_prompting(llm_model=model, prompt=query)
        except:
            # If the request fails, wait and retry once
            time.sleep(2)
            response = model_prompting(llm_model=model, prompt=query)

        return response

    def eval(self, prediction: str, ground_truth: str, metric: str) -> float:
        """
        Evaluate the model's prediction against the ground truth using the specified metric.

        Args:
            prediction: The model's output text
            ground_truth: The correct expected answer
            metric: The evaluation metric to use (e.g., 'em', 'f1_score', 'GSM8K')
            task_id: Optional identifier for the specific task being evaluated

        Returns:
            float: Evaluation score (typically between 0 and 1)
        """
        # Exact match evaluation
        if metric == 'em':
            result = exact_match_score(prediction, ground_truth)
            return float(result)

        # Multiple choice exact match
        elif metric == 'em_mc':
            result = exact_match_score(prediction, ground_truth, normal_method="mc")
            return float(result)

        # BERT-based semantic similarity score
        elif metric == 'bert_score':
            result = get_bert_score([prediction], [ground_truth])
            return result

        # GSM8K math problem evaluation
        # Extracts the final answer from the format "<answer>" and checks against ground truth
        elif metric == 'GSM8K':
            # Extract the final answer from ground truth (after the "####" delimiter)
            ground_truth = ground_truth.split("####")[-1].strip()

            # Look for an answer enclosed in angle brackets <X>
            match = re.search(r'\<(\d+)\>', prediction)
            if match:
                if match.group(1) == ground_truth:
                    return 1  # Correct answer
                else:
                    return 0  # Incorrect answer
            else:
                return 0  # No answer in expected format

        # F1 score for partial matching (used in QA tasks)
        elif metric == 'f1_score':
            f1, prec, recall = f1_score(prediction, ground_truth)
            return f1

        # Default case for unrecognized metrics
        else:
            return 0