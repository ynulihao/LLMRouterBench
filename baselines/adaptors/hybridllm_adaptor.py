"""
Hybrid LLM data adaptor for LLMRouterBench.

Converts baseline benchmark results to Hybrid LLM format (JSONL files).
This adaptor is specifically for Hybrid LLM (not BEST-Route) which uses
single sampling per model per prompt for pairwise model routing.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from .common import get_unique_models


class HybridLLMAdaptor:
    """
    Adaptor for converting baseline data to Hybrid LLM format.

    Hybrid LLM requires JSONL files with the following structure per line:
    {
        "id": "dataset_id/record_index",
        "instruction": "",
        "input": "prompt_text",
        "output": "ground_truth",
        "candidates": [
            {
                "model": "model_name",
                "text": "raw_output",
                "decoding_method": "greedy",  # or "sampling"
                "scores": {
                    "bartscore": score_value
                },
                "token_num_prompt": num_prompt_tokens,
                "token_num_responses": num_completion_tokens,
                "cost": cost_in_usd
            },
            ...
        ]
    }

    Key differences from BEST-Route:
    - Single sample per model (no best-of-n variants like bo1, bo2, etc.)
    - Focuses on pairwise routing between models
    - Uses actual evaluation scores from baseline data
    """

    def __init__(
        self,
        config_path: str = "config/baseline_config.yaml",
        random_seed: int = 42,
        train_ratio: float = 0.8,
        ood_datasets: Optional[List[str]] = None,
        selected_models: Optional[List[str]] = None
    ):
        """
        Initialize Hybrid LLM adaptor.

        Args:
            config_path: Path to baseline configuration YAML file
            random_seed: Random seed for train/test splitting
            train_ratio: Proportion of data for training (0.0-1.0)
            ood_datasets: Optional list of dataset IDs to treat as OOD (all go to test)
            selected_models: Must contain exactly 2 model names for pairwise routing

        Raises:
            ValueError: If selected_models is provided but doesn't contain exactly 2 models
        """
        # Validate selected_models for Hybrid LLM (pairwise routing)
        if selected_models is not None and len(selected_models) != 2:
            raise ValueError(
                f"Hybrid LLM requires exactly 2 models for pairwise routing, "
                f"but got {len(selected_models)}: {selected_models}"
            )

        self.config_path = config_path
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.ood_datasets = ood_datasets or []
        self.selected_models = selected_models
        self.loader = BaselineDataLoader(config_path=config_path)

        logger.info(f"Initialized Hybrid LLM adaptor with seed={random_seed}, ratio={train_ratio}")
        if self.ood_datasets:
            logger.info(f"OOD datasets: {self.ood_datasets}")
        if self.selected_models:
            logger.info(f"Model pair for pairwise routing: {self.selected_models}")

    def convert(self, output_dir: str = "baselines/Best-route-llm/data") -> Dict[str, str]:
        """
        Convert baseline data to Hybrid LLM format.

        Args:
            output_dir: Directory to save output JSONL files

        Returns:
            Dictionary mapping output type to file path

        Raises:
            ValueError: If selected_models is not specified (Hybrid LLM requires exactly 2 models)
        """
        # Enforce model pair requirement for Hybrid LLM
        if self.selected_models is None:
            raise ValueError(
                "Hybrid LLM requires exactly 2 models for pairwise routing. "
                "Please specify --models with exactly 2 model names, e.g., --models 'gpt-4,llama-3-8b'"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all records
        logger.info("Loading baseline records...")
        all_records = self.loader.load_all_records()
        logger.info(f"Loaded {len(all_records)} records")

        # Filter by selected models (guaranteed to be exactly 2 models)
        original_count = len(all_records)
        all_records = [r for r in all_records if r.model_name in self.selected_models]
        logger.info(f"Filtered to {len(all_records)} records from {original_count} "
                   f"for model pair: {self.selected_models}")

        # Check if we have data
        if not all_records:
            raise ValueError(f"No records found for models {self.selected_models}. "
                           f"Check your model names.")

        # Split by prompt (not by record) to avoid data leakage
        train_records, test_records = self._split_records_by_prompt(all_records)

        logger.info(f"Split: {len(train_records)} train records, {len(test_records)} test records")

        # Get unique models
        all_models = get_unique_models(all_records)
        logger.info(f"Found {len(all_models)} unique models: {all_models}")

        # Create output directory with seed and split ratio
        seed_str = f"seed{self.random_seed}"
        split_str = f"split{self.train_ratio:.1f}"
        output_path = output_path / f"hybridllm_{seed_str}_{split_str}"
        output_path.mkdir(parents=True, exist_ok=True)

        train_file = output_path / "train.jsonl"
        test_file = output_path / "test.jsonl"

        # Write train JSONL
        logger.info(f"Writing train data to {train_file}")
        self._write_jsonl(train_file, train_records)

        # Write test JSONL
        logger.info(f"Writing test data to {test_file}")
        self._write_jsonl(test_file, test_records)

        logger.info("Hybrid LLM conversion complete!")

        return {
            'train': str(train_file),
            'test': str(test_file)
        }

    def _split_records_by_prompt(
        self,
        all_records: List[BaselineRecord]
    ) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
        """
        Split records into train/test by dataset, then by prompts within each dataset.

        This ensures that:
        1. Each dataset is split independently
        2. All model evaluations for the same prompt stay together
        3. Prevents data leakage across prompts

        Uses BaselineDataLoader's unified splitting logic.

        Args:
            all_records: All baseline records

        Returns:
            Tuple of (train_records, test_records)
        """
        return self.loader.split_by_dataset_then_prompt(
            all_records,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            ood_datasets=self.ood_datasets
        )

    def _write_jsonl(
        self,
        file_path: Path,
        records: List[BaselineRecord]
    ):
        """
        Write records to JSONL file in Hybrid LLM format.

        Groups records by prompt and creates one JSON object per prompt with
        exactly 2 model candidates (fills missing data with zeros).

        Args:
            file_path: Output file path
            records: List of baseline records
        """
        from collections import Counter

        # Group records by (dataset_id, prompt) to get all model responses for same prompt
        prompt_groups = defaultdict(list)
        for record in records:
            key = (record.dataset_id, record.prompt, record.record_index)
            prompt_groups[key].append(record)

        logger.info(f"Writing {len(prompt_groups)} prompts to {file_path}")

        # Track statistics for filled missing data
        filled_counter = Counter()
        total_prompts = 0

        with open(file_path, 'w', encoding='utf-8') as f:
            for (dataset_id, prompt, record_index), group_records in prompt_groups.items():
                # Create model_name -> record mapping
                model_record_map = {r.model_name: r for r in group_records}

                # Get common fields from first available record
                sample_record = group_records[0]

                # Build candidates list - exactly 2 models (guaranteed by validation)
                candidates = []
                for model_name in self.selected_models:
                    if model_name in model_record_map:
                        # Use actual data
                        record = model_record_map[model_name]
                        candidate = {
                            "model": model_name,
                            "text": record.raw_output,
                            "decoding_method": "greedy",
                            "scores": {
                                "bartscore": float(record.score)
                            },
                            "token_num_prompt": record.prompt_tokens,
                            "token_num_responses": record.completion_tokens,
                            "cost": [float(record.cost)]
                        }
                    else:
                        # Fill missing data with zeros
                        candidate = {
                            "model": model_name,
                            "text": "",
                            "decoding_method": "greedy",
                            "scores": {
                                "bartscore": 0.0
                            },
                            "token_num_prompt": 0,
                            "token_num_responses": 0,
                            "cost": [0.0]
                        }
                        filled_counter[model_name] += 1

                    candidates.append(candidate)

                # Verify we have exactly 2 candidates
                assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"

                # Build the complete data point
                data_point = {
                    "id": f"{dataset_id}/{record_index}",
                    "instruction": "",  # Hybrid LLM format expects empty instruction
                    "input": prompt,
                    "output": sample_record.ground_truth,
                    "candidates": candidates
                }

                # Write as single line JSON
                f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
                total_prompts += 1

        logger.info(f"Successfully wrote {total_prompts} data points to {file_path}")

        # Log statistics about filled missing data
        if filled_counter:
            total_filled = sum(filled_counter.values())
            logger.info(f"Filled {total_filled} missing model evaluations:")
            for model_name in self.selected_models:
                count = filled_counter.get(model_name, 0)
                percentage = (count / total_prompts * 100) if total_prompts > 0 else 0
                logger.info(f"  - {model_name}: {count} prompts ({percentage:.1f}%)")
        else:
            logger.info("No missing data - all prompts have both model evaluations")


def main():
    """Main entry point for Hybrid LLM adaptor."""
    parser = argparse.ArgumentParser(
        description='Convert LLMRouterBench baseline data to Hybrid LLM format.\n'
                    'NOTE: Hybrid LLM uses pairwise routing, requiring EXACTLY 2 models.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/baseline_config.yaml',
        help='Path to baseline configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/test splitting'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (0.0-1.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='baselines/Best-route-llm/data',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--ood-datasets',
        type=str,
        default='',
        help='Comma-separated list of dataset IDs to treat as OOD (all go to test). '
             'Example: brainteaser,dailydialog'
    )
    parser.add_argument(
        '--models',
        type=str,
        required=True,
        help='REQUIRED: Comma-separated list of EXACTLY 2 model names for pairwise routing. '
             'Example: --models "gpt-4,llama-3-8b"'
    )

    args = parser.parse_args()

    # Parse OOD datasets
    ood_datasets = []
    if args.ood_datasets:
        ood_datasets = [d.strip() for d in args.ood_datasets.split(',') if d.strip()]

    # Parse and validate selected models (must be exactly 2)
    if not args.models:
        parser.error("--models is required and must specify exactly 2 model names")

    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]

    if len(selected_models) != 2:
        parser.error(
            f"Hybrid LLM requires exactly 2 models for pairwise routing, "
            f"but got {len(selected_models)}: {selected_models}\n"
            f"Example: --models 'gpt-4,llama-3-8b'"
        )

    # Create adaptor and convert
    adaptor = HybridLLMAdaptor(
        config_path=args.config,
        random_seed=args.seed,
        train_ratio=args.split_ratio,
        ood_datasets=ood_datasets,
        selected_models=selected_models
    )

    output_files = adaptor.convert(output_dir=args.output_dir)

    logger.info("Generated files:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")

    logger.info("\nTo use these files with Hybrid LLM:")
    logger.info(f"  Train data: {output_files['train']}")
    logger.info(f"  Test data: {output_files['test']}")
    logger.info(f"  Model pair: {selected_models}")
    logger.info("\nExample training command:")
    logger.info("  deepspeed --num_gpus=8 \\")
    logger.info("    train_router_gte.py \\")
    logger.info(f"    --train_data_path {output_files['train']} \\")
    logger.info(f"    --test_data_path {output_files['test']} \\")
    logger.info(f"    --eval_data_path {output_files['test']} \\")
    logger.info("    --do_eval True \\")
    logger.info("    --evaluation_strategy steps \\")
    logger.info("    --eval_steps 5 \\")
    logger.info("    --save_strategy steps \\")
    logger.info("    --save_steps 50 \\")
    logger.info(f"    --candidate_models {','.join(selected_models)} \\")
    logger.info("    --per_device_train_batch_size 1 \\")
    logger.info("    --per_device_eval_batch_size 1 \\")
    logger.info("    --gradient_accumulation_steps 64 \\")
    logger.info("    --fp16 False \\")
    logger.info("    --deepspeed config/ds_zero2.json \\")
    logger.info(f"    --output_dir outputs/seed{args.seed}_split{args.split_ratio} \\")
    logger.info(f"    --run_name seed{args.seed}_split{args.split_ratio} \\")
    logger.info("    --num_train_epochs 5")


if __name__ == '__main__':
    main()
