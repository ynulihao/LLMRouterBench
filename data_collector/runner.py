import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

from .config_loader import BenchmarkConfig
from .planner import RunPlan
from .storage import ResultsStorage, BenchmarkResult, RecordResult
from evaluation.factory import EvaluatorFactory
from generators.factory import create_generator


def extract_field_from_response(response_json: str, path: str) -> Any:
    """
    Extract a field from response JSON string by path.

    Supported path formats:
    - "usage.prompt_tokens" -> response["usage"]["prompt_tokens"]
    - "choices[0].message.content" -> response["choices"][0]["message"]["content"]

    Args:
        response_json: API response JSON string
        path: Field path string

    Returns:
        Extracted value, or None if extraction fails
    """
    if not response_json or not path:
        return None

    try:
        response = json.loads(response_json)
    except json.JSONDecodeError:
        return None

    current = response
    # Split path: "choices[0].message.content" -> ["choices", "0", "message", "content"]
    tokens = re.split(r'\.|\[|\]', path)
    tokens = [t for t in tokens if t]  # Remove empty strings

    try:
        for token in tokens:
            if token.isdigit():
                # Index access
                current = current[int(token)]
            elif isinstance(current, dict):
                current = current[token]
            else:
                return None
        return current
    except (KeyError, IndexError, TypeError):
        return None


def extract_extra_fields(response_json: str, extract_config: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract multiple fields from response based on config.

    Args:
        response_json: API response JSON string
        extract_config: {field_name: path} dictionary

    Returns:
        {field_name: value} dictionary, only contains successfully extracted fields
    """
    if not response_json or not extract_config:
        return {}

    result = {}
    for field_name, path in extract_config.items():
        value = extract_field_from_response(response_json, path)
        if value is not None:
            result[field_name] = value
    return result


class BenchmarkRunner:
    """Execute benchmark runs with concurrent processing"""

    def __init__(self, config: BenchmarkConfig, storage: ResultsStorage):
        self.config = config
        self.storage = storage
        self.evaluator_factory = EvaluatorFactory(grader_cache_config=config.grader_cache_config)
    
    def run_all(self, plans: List[RunPlan]) -> Dict[str, Any]:
        """Execute all planned runs"""
        if not plans:
            logger.info("No runs to execute")
            return {"total_runs": 0, "successful_runs": 0, "failed_runs": 0}
        
        if self.config.run.demo_mode:
            logger.info(f"Starting execution in DEMO MODE - limiting each dataset to {self.config.run.demo_limit} records")
            logger.info(f"Demo mode: {len(plans)} runs planned (results will not be indexed)")
        else:
            logger.info(f"Starting execution of {len(plans)} runs")
        start_time = time.time()
        
        results = {
            "total_runs": len(plans),
            "successful_runs": 0,
            "failed_runs": 0,
            "run_details": []
        }
        
        for plan in plans:
            logger.info(f"Processing run: {plan.run_key}")
            
            try:
                run_result = self.execute_single_run(plan)
                if run_result:
                    results["successful_runs"] += 1
                    results["run_details"].append({
                        "run_key": plan.run_key,
                        "status": "success",
                        "performance": run_result.performance,
                        "time_taken": run_result.time_taken
                    })
                else:
                    results["failed_runs"] += 1
                    results["run_details"].append({
                        "run_key": plan.run_key,
                        "status": "failed",
                        "error": "Unknown error"
                    })
            except Exception as e:
                logger.error(f"Failed to execute run {plan.run_key}: {str(e)}")
                results["failed_runs"] += 1
                results["run_details"].append({
                    "run_key": plan.run_key,
                    "status": "failed",
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Completed {results['successful_runs']}/{results['total_runs']} runs in {total_time:.2f}s")
        
        return results
    
    def execute_single_run(self, plan: RunPlan) -> BenchmarkResult:
        """Execute a single benchmark run"""
        start_time = time.time()
        
        try:
            # 1. Get evaluator for this dataset
            evaluator = self.evaluator_factory.get_evaluator(plan.dataset_id)
            
            # 2. Load data (format_prompt already done in load_data)
            data = evaluator.load_data(plan.split)

            if not data:
                raise ValueError(f"No data loaded for {plan.dataset_id}/{plan.split}")

            # Calculate data fingerprint (before any modifications)
            data_fingerprint = self.storage.calculate_data_fingerprint(data)

            # Demo mode: limit data size
            if self.config.run.demo_mode:
                original_size = len(data)
                # Handle different data types properly
                if hasattr(data, 'select'):  # Dataset object
                    data = data.select(range(min(self.config.run.demo_limit, len(data))))
                else:  # List or other sequence
                    data = data[:self.config.run.demo_limit]
                logger.info(f"Demo mode: limited to {len(data)} records (from {original_size} total)")
            else:
                logger.info(f"Loaded {len(data)} records for {plan.dataset_id}/{plan.split}")
            
            # 3. Get model configuration
            model_config = self._get_model_config(plan.model_name)
            if not model_config:
                raise ValueError(f"Model configuration not found: {plan.model_name}")
            
            # 4. Create generator
            generator = create_generator(
                model_config=model_config.__dict__,
                cache_config=self.config.cache_config
            )
            logger.info(f"Created generator: {type(generator).__name__} for model {plan.model_name}")
            
            # 5. Process records concurrently
            records = self._process_records_concurrent(
                data=data,
                generator=generator,
                evaluator=evaluator,
                concurrency=self.config.run.concurrency,
                model_config=model_config
            )
            
            # 6. Calculate aggregated statistics
            performance, total_prompt_tokens, total_completion_tokens, total_cost = self._calculate_aggregates(records)
            
            # 7. Create result object
            result = BenchmarkResult(
                performance=performance,
                time_taken=time.time() - start_time,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cost=total_cost,
                counts=len(records),
                model_name=plan.model_name,
                dataset_name=plan.dataset_id,
                split=plan.split,
                demo=self.config.run.demo_mode,
                records=records
            )
            
            # 8. Save result
            self.storage.save_result(result, plan.dataset_id, plan.split, plan.model_name, data_fingerprint)

            logger.info(f"Completed run {plan.run_key}: {performance:.3f} performance")
            return result
            
        except Exception as e:
            logger.error(f"Error executing run {plan.run_key}: {str(e)}")
            raise
    
    def _process_records_concurrent(self, data: List[Dict[str, Any]], generator, evaluator, concurrency: int, model_config=None) -> List[RecordResult]:
        """Process records with concurrent execution"""
        from tqdm import tqdm

        def process_single_record(record_data: Dict[str, Any], index: int) -> RecordResult:
            """Process a single record"""
            try:
                # Extract required fields
                origin_query = record_data.get('origin_query', record_data.get('question', ''))
                prompt = record_data.get('prompt', record_data.get('formatted_prompt', ''))
                
                if not prompt:
                    raise ValueError("No prompt found in record")
                
                # Generate response (with images for multimodal generators)
                images = record_data.get('image_paths', [])
                if images:
                    gen_output = generator.generate(prompt, images=images)
                else:
                    gen_output = generator.generate(prompt)
                
                # Evaluate response using the correct interface
                raw_output = gen_output.output
                eval_result = evaluator.evaluate(record_data, raw_output)
                
                # Extract evaluation results
                # Handle boolean, decimal, and None is_correct values
                is_correct_value = eval_result.get('is_correct', False)
                if is_correct_value is None:
                    # Handle evaluators without internal scoring (e.g., ArenaHard)
                    score = None
                elif isinstance(is_correct_value, bool):
                    score = 1.0 if is_correct_value else 0.0
                else:
                    # Handle decimal values (0-1 range)
                    score = float(is_correct_value)
                prediction = eval_result.get('prediction', '')
                ground_truth = eval_result.get('ground_truth', '')

                # Extract extra fields from raw response if configured
                extra_fields = {}
                if model_config and model_config.extract_fields and gen_output.raw_response:
                    extra_fields = extract_extra_fields(
                        gen_output.raw_response,
                        model_config.extract_fields
                    )

                return RecordResult(
                    index=index + 1,  # 1-based indexing
                    origin_query=origin_query,
                    prompt=prompt,
                    prompt_tokens=gen_output.prompt_tokens,
                    completion_tokens=gen_output.completion_tokens,
                    cost=gen_output.cost,
                    score=score,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    raw_output=raw_output,
                    extra_fields=extra_fields
                )
                
            except Exception as e:
                logger.warning(f"Failed to process record {index}: {str(e)}")
                return RecordResult(
                    index=index + 1,
                    origin_query=record_data.get('origin_query', record_data.get('question', '')),
                    prompt=record_data.get('prompt', record_data.get('formatted_prompt', '')),
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost=0.0,
                    score=None,
                    prediction="",
                    ground_truth="",
                    raw_output=f"Processing failed: {str(e)}"
                )
        
        # Execute concurrently with progress tracking
        results = [None] * len(data)
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_record, record, idx): idx 
                for idx, record in enumerate(data)
            }
            
            # Collect results with progress bar
            with tqdm(total=len(data), desc="Processing records", unit="record") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        logger.error(f"Unexpected error processing record {idx}: {e}")
                        results[idx] = RecordResult(
                            index=idx + 1,
                            origin_query="",
                            prompt="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost=0.0,
                            score=None,
                            prediction="",
                            ground_truth="",
                            raw_output=f"Unexpected error: {str(e)}"
                        )
                    
                    # Update progress bar
                    pbar.update(1)
        
        return results
    
    def _calculate_aggregates(self, records: List[RecordResult]) -> tuple[float, int, int, float]:
        """Calculate aggregated statistics from records"""
        total_prompt_tokens = sum(r.prompt_tokens for r in records)
        total_completion_tokens = sum(r.completion_tokens for r in records)
        total_cost = sum(r.cost for r in records)
        
        # Calculate performance (average of non-null scores)
        valid_scores = [r.score for r in records if r.score is not None]
        performance = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return performance, total_prompt_tokens, total_completion_tokens, total_cost
    
    def _get_model_config(self, model_name: str):
        """Get model configuration by name"""
        for model in self.config.models:
            if model.name == model_name:
                return model
        return None
    
