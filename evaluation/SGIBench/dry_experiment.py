"""
SGI-Bench Dry Experiment Evaluator

Task 3.1: Code completion for scientific computing.
Evaluates by executing generated code and comparing outputs.

IMPORTANT: This evaluator requires:
1. The 'dryexp' conda environment (official SGI-Bench requirement)
2. Benchmark data fixtures for certain samples (SGI_DryExperiment_0200, 0206, 0236)

Set SGI_DRY_EXPERIMENT_DATA_DIR to point to the data directory containing:
- SGI_DryExperiment_0200/
- SGI_DryExperiment_0206/
- SGI_DryExperiment_0236/
"""

import os
import re
import json
import subprocess
import tempfile
import shutil
import time
from typing import Dict, Any, List, Optional

from datasets import Dataset
from loguru import logger

from evaluation.SGIBench.base import SGIBenchBaseEvaluator
from evaluation.SGIBench.prompts import (
    DRY_EXPERIMENT_OUTPUT_REQUIREMENTS,
    DRY_EXPERIMENT_JUDGE_PROMPT
)
from evaluation.SGIBench.utils import (
    check_syntax,
    replace_function,
    AnswerParser
)

# Answer example for LLM parser (per official step_2_get_answer.py lines 44-50)
CODE_ANSWER_EXAMPLE = """
def add(a, b):
    return a+b

def minus(a, b):
    return a-b
"""


class DryExperimentEvaluator(SGIBenchBaseEvaluator):
    """
    Evaluator for SGI-Bench Dry Experiment task.

    This task evaluates the model's ability to complete Python functions
    for scientific computing. Evaluation is done by:
    1. Replacing incomplete functions with generated code
    2. Executing 5 unit tests per sample
    3. Comparing outputs:
       - Exact string match first
       - LLM judge for non-exact matches (strict digit accuracy,
         only training loss/metrics allow 2% tolerance)

    Metrics:
    - PassAll@5: All 5 tests pass
    - PassAll@3: At least 3 tests pass
    - PassAll@1: At least 1 test passes
    - AET: Average Execution Time
    - SER: Smooth Execution Rate (successful executions / 5)

    Requirements:
    - conda environment 'dryexp' must exist (override with SGI_CONDA_ENV)
    - For samples 0200, 0206, 0236: data fixtures must be available
    """

    # Number of unit tests per sample
    NUM_UNIT_TESTS = 5

    # Samples that require additional data fixtures (from official step_1_build.py)
    # Format: {sample_idx: (source_subdir, target_subdir)}
    FIXTURE_SAMPLES = {
        "SGI_DryExperiment_0200": ("SGI_DryExperiment_0200", "data"),
        "SGI_DryExperiment_0206": ("SGI_DryExperiment_0206", "data/mnist_raw"),
        "SGI_DryExperiment_0236": ("SGI_DryExperiment_0236", "data/em_3d_user_study"),
    }

    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize DryExperimentEvaluator.

        Args:
            grader_cache_config: Optional cache configuration for LLM grader

        Raises:
            RuntimeError: If the required conda environment does not exist
        """
        super().__init__(grader_cache_config)
        self.task = "SGIBench-DryExperiment"
        self.data_path = os.path.join(self.DATA_DIR, "dry_experiment", "test.jsonl")

        # Conda environment (default: dryexp per official implementation)
        self.conda_env = os.getenv("SGI_CONDA_ENV", "dryexp")
        self.timeout = int(os.getenv("SGI_EXECUTION_TIMEOUT", "300"))  # 5 minutes

        # Data fixtures directory
        self.fixture_data_dir = os.getenv("SGI_DRY_EXPERIMENT_DATA_DIR", "")

        # Validate conda environment exists
        self._validate_conda_env()

        # Initialize grader with SGI-Bench Dry Experiment specific config
        # Official: judge = LLM('o4-mini') with temperature=0
        self._init_sgi_grader(
            model_env_var="SGI_DE_GRADER_MODEL",
            default_model="o4-mini",
            temperature=0.0
        )

        # Initialize answer parser for LLM fallback (per official step_2_get_answer.py)
        self.answer_parser = AnswerParser(generator=self.grader)

    def _validate_conda_env(self):
        """
        Validate that the required conda environment exists.

        Raises:
            RuntimeError: If the conda environment does not exist
        """
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to list conda environments: {result.stderr}"
                )

            # Check if environment exists in the list
            env_list = result.stdout
            if self.conda_env not in env_list:
                raise RuntimeError(
                    f"Required conda environment '{self.conda_env}' does not exist.\n"
                    f"Please create it using the official SGI-Bench setup:\n"
                    f"  conda create -n {self.conda_env} python=3.10\n"
                    f"  conda activate {self.conda_env}\n"
                    f"  # Install required packages for SGI-Bench dry experiment\n"
                    f"\n"
                    f"Or set SGI_CONDA_ENV to use a different environment."
                )

            logger.info(f"Validated conda environment: {self.conda_env}")

        except FileNotFoundError:
            raise RuntimeError(
                "conda command not found. Please ensure conda is installed and in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout while checking conda environments.")

    def _prepare_fixtures(self, sample_idx: str, work_dir: str) -> bool:
        """
        Copy required data fixtures for specific samples.

        Args:
            sample_idx: Sample identifier (e.g., "SGI_DryExperiment_0200")
            work_dir: Working directory for execution

        Returns:
            True if fixtures were copied or not needed, False if required but missing
        """
        if sample_idx not in self.FIXTURE_SAMPLES:
            return True  # No fixtures needed for this sample

        source_subdir, target_subdir = self.FIXTURE_SAMPLES[sample_idx]

        if not self.fixture_data_dir:
            logger.warning(
                f"Sample {sample_idx} requires data fixtures but SGI_DRY_EXPERIMENT_DATA_DIR is not set. "
                f"This sample may fail to execute correctly."
            )
            return False

        source_path = os.path.join(self.fixture_data_dir, source_subdir)
        target_path = os.path.join(work_dir, target_subdir)

        if not os.path.exists(source_path):
            logger.warning(
                f"Data fixtures for {sample_idx} not found at {source_path}. "
                f"This sample may fail to execute correctly."
            )
            return False

        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            logger.debug(f"Copied fixtures from {source_path} to {target_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to copy fixtures for {sample_idx}: {e}")
            return False

    def load_data(self, split: str = "test") -> Dataset:
        """
        Load dry experiment data.

        Args:
            split: Data split (only "test" is supported)

        Returns:
            Dataset with formatted prompts
        """
        data = self.load_jsonl(self.data_path)
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.format_prompt)
        return dataset

    def get_valid_splits(self) -> List[str]:
        """Get valid data splits."""
        return ["test"]

    def format_prompt(self, item: Dict) -> Dict:
        """
        Format the prompt for dry experiment task.

        Args:
            item: Data item containing 'question' field

        Returns:
            Item with 'prompt' field added
        """
        question = item.get("question", "")
        prompt = question + DRY_EXPERIMENT_OUTPUT_REQUIREMENTS

        return {
            **item,
            "prompt": prompt,
            "origin_query": question
        }

    def _run_code(self, code: str, data_code: str, work_dir: str) -> Dict:
        """
        Execute Python code and capture output.

        Args:
            code: Main code to execute
            data_code: Data generation code
            work_dir: Working directory for execution

        Returns:
            Dictionary with execution results
        """
        result = {
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "runtime": -1,
            "error": ""
        }

        try:
            # Write code files
            main_path = os.path.join(work_dir, "main_en.py")
            data_path = os.path.join(work_dir, "data_en.py")

            with open(main_path, 'w', encoding='utf-8') as f:
                f.write(code)

            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(data_code)

            # Set up environment
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Step 1: Run data_en.py to generate data files
            # Use 10 minutes timeout for data generation (per official step_1_build.py)
            data_timeout = max(self.timeout, 600)  # At least 10 minutes
            data_cmd = ["conda", "run", "-n", self.conda_env, "python", "data_en.py"]
            data_proc = subprocess.run(
                data_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=data_timeout,
                encoding="utf-8",
                env=env
            )

            if data_proc.returncode != 0:
                result["error"] = f"[WRONG] data_en.py failed: {data_proc.stderr}"
                return result

            # Step 2: Run main_en.py
            cmd = ["conda", "run", "-n", self.conda_env, "python", "main_en.py"]
            start_time = time.time()
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                env=env
            )
            end_time = time.time()

            result["returncode"] = proc.returncode
            result["stdout"] = proc.stdout.strip()
            result["stderr"] = proc.stderr.strip()
            result["runtime"] = end_time - start_time

            if proc.returncode != 0:
                result["error"] = f"[WRONG] {proc.stderr}" if proc.stderr else "Unknown error"

        except subprocess.TimeoutExpired:
            result["error"] = f"[WRONG] Execution timed out after {self.timeout}s"
            result["runtime"] = self.timeout

        except Exception as e:
            result["error"] = f"[WRONG] {str(e)}"

        return result

    def _compare_outputs(self, expected: str, actual: str) -> Dict:
        """
        Compare expected and actual outputs.

        Per official implementation:
        1. First tries exact string match
        2. If not exact match, uses LLM judge with strict rules:
           - All numerical values must match exactly (every digit)
           - Only training-related loss/metrics allow 2% tolerance

        Args:
            expected: Expected output
            actual: Actual output from code execution

        Returns:
            Dictionary with comparison result
        """
        # Exact string match (official: line 31)
        if expected == actual:
            return {
                "pass": True,
                "method": "exact_match",
                "llm_judge": {"judgment": "correct", "reason": "Exact match."}
            }

        # Use LLM judge for non-exact matches (official: lines 45-73)
        try:
            judge_prompt = DRY_EXPERIMENT_JUDGE_PROMPT.format(
                expected_output=expected,
                model_output=actual
            )

            judge_result = self.grader.generate(judge_prompt)
            output = judge_result.output

            # Update token counts
            self.prompt_tokens += judge_result.prompt_tokens
            self.completion_tokens += judge_result.completion_tokens

            # Parse judgment as JSON (official: lines 66-68)
            try:
                # Try to find JSON in output
                start_index = output.find('{')
                end_index = output.rfind('}') + 1

                if start_index != -1 and end_index > start_index:
                    json_str = output[start_index:end_index]

                    # Try json_repair first, fallback to standard json
                    try:
                        from json_repair import repair_json
                        llm_judge = eval(repair_json(json_str))
                    except ImportError:
                        llm_judge = json.loads(json_str)

                    # Strict equality check: judgment must be exactly "correct"
                    # (NOT substring match - "incorrect" contains "correct"!)
                    is_pass = llm_judge.get("judgment", "").lower() == "correct"

                    return {
                        "pass": is_pass,
                        "method": "llm_judge",
                        "llm_judge": llm_judge
                    }
                else:
                    logger.warning("Could not find JSON in LLM judge output")
                    return {"pass": False, "method": "llm_judge_parse_failed", "llm_judge": None}

            except Exception as parse_error:
                logger.warning(f"Failed to parse LLM judge output: {parse_error}")
                return {"pass": False, "method": "llm_judge_parse_failed", "llm_judge": None}

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return {"pass": False, "method": "failed", "error": str(e)}

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        """
        Evaluate model output by executing generated code.

        Evaluation process:
        1. Extract code from model output (between <answer> tags)
        2. Replace incomplete functions in main code
        3. Execute 5 unit tests
        4. Compare outputs and calculate metrics

        Args:
            data: Input data item containing code and expected outputs
            output_text: Model output text

        Returns:
            Evaluation result dictionary with various metrics
        """
        # Extract code from model output (official: extract_final_answer)
        extracted_code = self.extract_sgi_answer(output_text)

        if extracted_code is None:
            extracted_code = output_text

        # Get LLM-parsed answer as fallback (per official step_2_get_answer.py lines 77-81)
        parsed_code = None
        try:
            parsed_code = self.answer_parser.parse(extracted_code, CODE_ANSWER_EXAMPLE)
            if parsed_code:
                logger.debug(f"LLM parser produced fallback code")
        except Exception as e:
            logger.debug(f"LLM parser failed: {e}")

        # Get incomplete function names and main code
        incomplete_functions = data.get("incomplete_functions", [])
        incomplete_main_code = data.get("incomplete_main_code", "")

        # Get sample idx for fixture handling
        sample_idx = data.get("idx", "")

        # Try to replace functions (per official step_2_get_answer.py lines 83-90)
        # For each function: try with extracted_code first, then with parsed_code as fallback
        main_code = incomplete_main_code
        replacement_success = False

        for func_name in incomplete_functions:
            replaced = False
            # Try with extracted code first
            if check_syntax(extracted_code):
                try:
                    main_code = replace_function(main_code, extracted_code, func_name)
                    replaced = True
                    replacement_success = True
                except Exception:
                    pass

            # Fallback to LLM-parsed code if first attempt failed
            if not replaced and parsed_code and check_syntax(parsed_code):
                try:
                    main_code = replace_function(main_code, parsed_code, func_name)
                    replacement_success = True
                    logger.debug(f"Used LLM-parsed code for function {func_name}")
                except Exception as e:
                    logger.warning(f"Failed to replace function {func_name}: {e}")

        if not replacement_success:
            # Code replacement failed
            return {
                "prediction": extracted_code,
                "ground_truth": data.get("main_code", ""),
                "is_correct": False,
                "PassAll@5": 0,
                "PassAll@3": 0,
                "PassAll@1": 0,
                "AET": -1,
                "SER": 0.0,
                "error": "Failed to replace functions in main code",
                "unit_test_results": []
            }

        # Run unit tests
        unit_test_results = []
        pass_count = 0
        total_runtime = 0
        successful_executions = 0

        for i in range(self.NUM_UNIT_TESTS):
            # Get data code and expected output for this test
            data_code_key = f"unit_test_{i}_data"
            output_key = f"unit_test_{i}_output"

            data_code = data.get(data_code_key, data.get("data_code", ""))
            expected_output = data.get(output_key, "")

            # Create temporary directory for execution
            with tempfile.TemporaryDirectory() as work_dir:
                # Create data subdirectory (required by official flow)
                os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)

                # Copy fixtures if needed for this sample
                self._prepare_fixtures(sample_idx, work_dir)

                # Run the code
                exec_result = self._run_code(main_code, data_code, work_dir)

                test_result = {
                    "unit_test_idx": i,
                    "returncode": exec_result["returncode"],
                    "runtime": exec_result["runtime"],
                    "error": exec_result["error"],
                    "model_output": f"{exec_result['stderr']}\n{exec_result['stdout']}".strip(),
                    "expected_output": expected_output
                }

                # Check if execution was successful
                if exec_result["returncode"] == 0:
                    successful_executions += 1
                    total_runtime += exec_result["runtime"]

                    # Compare outputs
                    comparison = self._compare_outputs(
                        expected_output,
                        test_result["model_output"]
                    )
                    test_result["pass"] = comparison["pass"]
                    test_result["comparison_method"] = comparison.get("method", "")

                    if comparison["pass"]:
                        pass_count += 1
                else:
                    test_result["pass"] = False
                    test_result["comparison_method"] = "execution_failed"

                unit_test_results.append(test_result)

        # Calculate metrics
        aet = total_runtime / successful_executions if successful_executions > 0 else -1
        ser = successful_executions / self.NUM_UNIT_TESTS

        return {
            "prediction": extracted_code,
            "ground_truth": data.get("main_code", ""),
            "is_correct": pass_count == self.NUM_UNIT_TESTS,
            "PassAll@5": 1 if pass_count == 5 else 0,
            "PassAll@3": 1 if pass_count >= 3 else 0,
            "PassAll@1": 1 if pass_count >= 1 else 0,
            "pass_count": pass_count,
            "AET": aet,
            "SER": ser,
            "unit_test_results": unit_test_results
        }
