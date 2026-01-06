# Adaptor

**LLMRouterBench Adaptor System - Data Loading, Transformation & Algorithm Integration**

This package provides efficient loading, transformation, and analysis of benchmark results for baseline comparisons across datasets and models.

---

## 📑 Table of Contents

- [**Part 1: Core API Reference**](#part-1-core-api-reference)
  - [1.1 BaselineDataLoader](#11-baselinedataloader)
  - [1.2 BaselineAggregator](#12-baselineaggregator)
  - [1.3 Data Schema](#13-data-schema)
  - [1.4 Quick Start Examples](#14-quick-start-examples)
- [**Part 2: Routing Algorithms & Adaptors**](#part-2-routing-algorithms--adaptors)
  - [2.1 Supported Algorithms](#21-supported-algorithms)
  - [2.2 Algorithm Quick Start](#22-algorithm-quick-start)
  - [2.3 Integrating New Algorithms](#23-integrating-new-algorithms)

---

# Part 1: Core API Reference

## 1.1 BaselineDataLoader

`BaselineDataLoader` provides efficient data loading with configurable filtering and multiple output formats.

### Constructor

```python
from baselines import BaselineDataLoader

loader = BaselineDataLoader(
    config_path='config/baseline_config.yaml'
)
```

**Parameters:**
- `config_path` (str, optional): Path to configuration YAML file
- `config` (dict, optional): Configuration dictionary (alternative to config_path)
- `include_reference_models` (bool, default=False): Whether to load reference models alongside main models

---

### Methods

#### `load_records_iter()`

Iterate over all baseline records (memory-efficient).

```python
for record in loader.load_records_iter():
    print(f"{record.dataset_id} - {record.model_name}: {record.score}")
```

**Returns:** `Iterator[BaselineRecord]` - Records yielded one at a time

**Use Case:** Large datasets where loading all records at once would consume too much memory.

---

#### `load_all_records()`

Load all records into memory as a list.

```python
all_records = loader.load_all_records()
print(f"Loaded {len(all_records)} records")
```

**Returns:** `List[BaselineRecord]` - All records in a list

**Warning:** May consume significant memory for large datasets. Consider using `load_records_iter()` for datasets with >100K records.

---

#### `to_dict_list()`

Convert all records to a list of dictionaries.

```python
dict_list = loader.to_dict_list(compact=True)
# [{'dataset_id': 'aime', 'model_name': 'gpt-4', ...}, ...]
```

**Parameters:**
- `compact` (bool, default=False): Use column selection from config if True

**Returns:** `List[Dict[str, Any]]` - List of dictionaries

---

#### `to_dataframe()`

Convert all records to a pandas DataFrame.

```python
import pandas as pd

df = loader.to_dataframe()
print(df.head())
print(df.shape)
```

**Returns:** `pd.DataFrame` - DataFrame with all baseline records

**Raises:** `ImportError` if pandas is not installed

---

#### `split_by_dataset_then_prompt()`

Split records into train and test sets while preventing data leakage.

```python
train_records, test_records = loader.split_by_dataset_then_prompt(
    records=all_records,
    train_ratio=0.8,
    random_seed=42,
    ood_datasets=['brainteaser', 'dailydialog']
)

print(f"Train: {len(train_records)}, Test: {len(test_records)}")
```

**Parameters:**
- `records` (List[BaselineRecord]): Records to split
- `train_ratio` (float, default=0.8): Proportion of prompts for training (0.0-1.0)
- `random_seed` (int, default=42): Random seed for reproducibility
- `ood_datasets` (List[str], optional): Dataset IDs to treat as out-of-distribution (all go to test)

**Returns:** `Tuple[List[BaselineRecord], List[BaselineRecord]]` - (train_records, test_records)

**Key Features:**
1. Groups records by dataset first, then by prompt
2. Each prompt appears in EITHER train OR test, not both
3. All model evaluations for the same prompt stay together
4. OOD datasets are entirely placed in test set
5. Ensures each dataset has representation in both sets (unless OOD)

**This prevents data leakage** - critical for training routing models.

---

## 1.2 BaselineAggregator

`BaselineAggregator` computes statistics and comparisons from baseline records.

### Constructor

```python
from baselines import BaselineAggregator

aggregator = BaselineAggregator(
    records=all_records,
    data_loader=loader  # optional, for test_mode and reference_models
)
```

**Parameters:**
- `records` (List[BaselineRecord]): Records to aggregate
- `data_loader` (BaselineDataLoader, optional): Loader instance for advanced features

---

### Methods

#### `get_global_stats()`

Compute global statistics across all records.

```python
stats = aggregator.get_global_stats()
print(f"Overall accuracy: {stats['avg_score']:.2%}")
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Datasets: {stats['datasets']}")
print(f"Models: {stats['models']}")
```

**Returns:** `Dict[str, Any]` with keys:
- `total_records`: Total number of records
- `total_datasets`: Number of unique datasets
- `total_models`: Number of unique models
- `avg_score`: Average score across all records
- `total_cost`: Total API cost
- `avg_cost_per_record`: Average cost per record
- `total_prompt_tokens`: Total prompt tokens
- `total_completion_tokens`: Total completion tokens
- `datasets`: List of dataset IDs
- `models`: List of model names

---

#### `aggregate_by_dataset_and_model()`

Aggregate statistics by dataset/split and model.

```python
stats_by_dataset_model = aggregator.aggregate_by_dataset_and_model()
# {'aime/test': {'gpt-4': AggregatedStats(...), ...}, ...}

gpt4_aime_stats = stats_by_dataset_model['aime/test']['gpt-4']
print(f"GPT-4 on AIME: {gpt4_aime_stats.avg_score:.2%}")
```

**Returns:** `Dict[str, Dict[str, AggregatedStats]]`
- Key: `"{dataset_id}/{split}"` (e.g., `"aime/test"`)
- Value: Dict mapping `model_name` to `AggregatedStats`

---

#### `aggregate_by_model()`

Aggregate statistics grouped by model across all datasets.

```python
stats_by_model = aggregator.aggregate_by_model()
# {'gpt-4': [AggregatedStats(...), AggregatedStats(...), ...], ...}

for stats in stats_by_model['gpt-4']:
    print(f"GPT-4 on {stats.dataset_id}: {stats.avg_score:.2%}")
```

**Returns:** `Dict[str, List[AggregatedStats]]`
- Key: `model_name`
- Value: List of `AggregatedStats` (one per dataset)

---

#### `aggregate_by_dataset()`

Aggregate statistics grouped by dataset across all models.

```python
stats_by_dataset = aggregator.aggregate_by_dataset()
# {'aime/test': [AggregatedStats(...), AggregatedStats(...), ...], ...}

for stats in stats_by_dataset['aime/test']:
    print(f"{stats.model_name} on AIME: {stats.avg_score:.2%}")
```

**Returns:** `Dict[str, List[AggregatedStats]]`
- Key: `"{dataset_id}/{split}"`
- Value: List of `AggregatedStats` (one per model)

---

#### `to_summary_table()`

Create comprehensive performance and cost pivot tables.

```python
perf_table, cost_table = aggregator.to_summary_table(
    cost_metric='total_cost',
    test_mode=False,
    random_seed=42,
    train_ratio=0.8,
    ood_datasets=None
)

print("Performance Table:")
print(perf_table)
print("\nCost Table:")
print(cost_table)
```

**Parameters:**
- `cost_metric` (str, default='total_cost'): Cost metric to use ('total_cost' or 'avg_cost_per_record')
- `test_mode` (bool, default=False): If True, compute statistics only for test set
- `random_seed` (int, default=42): Random seed for Random Router sampling
- `train_ratio` (float, default=0.8): Training proportion (used when test_mode=True)
- `ood_datasets` (List[str], optional): OOD dataset IDs (used when test_mode=True)

**Returns:** `Tuple[pd.DataFrame, pd.DataFrame]` - (performance_table, cost_table)

**Performance Table Rows:**
- Individual model names (sorted alphabetically)
- `AVG`: Average across main models
- `Random Router`: Simulated random selection performance
- `Max Expert`: Best model performance per dataset
- `Oracle`: Best possible performance with perfect routing

**Cost Table Rows:**
- Individual model names
- `AVG`: Average cost across main models
- `Random Router`: Cost when randomly selecting models
- `Max Expert`: Cost of best performing model per dataset
- `Oracle`: Minimum achievable cost with perfect routing

**Columns:**
- Dataset/split names
- Aggregate columns: `Dataset-Avg`, `Sample-Avg`, etc.

---

#### `print_summary_tables()`

Print formatted summary tables to console.

```python
aggregator.print_summary_tables(
    cost_metric='avg_cost_per_record',
    random_seed=42,
    decimal_places=4
)
```

**Parameters:**
- `cost_metric` (str, default='total_cost'): Cost metric
- `test_mode` (bool, default=False): Compute on test set only
- `random_seed` (int, default=42): Random seed
- `train_ratio` (float, default=0.8): Training proportion
- `ood_datasets` (List[str], optional): OOD datasets
- `decimal_places` (int, default=4): Number of decimal places for display

**Output:** Formatted tables printed to console with color highlighting

---

#### `save_summary_tables_to_excel()`

Export summary tables to an Excel file with formatting.

```python
aggregator.save_summary_tables_to_excel(
    output_path='results/summary_tables.xlsx',
    cost_metric='total_cost',
    random_seed=42
)
```

**Parameters:**
- `output_path` (str): Path for the Excel file
- `cost_metric` (str, default='total_cost'): Cost metric
- `test_mode` (bool, default=False): Compute on test set only
- `random_seed` (int, default=42): Random seed
- `train_ratio` (float, default=0.8): Training proportion
- `ood_datasets` (List[str], optional): OOD datasets
- `decimal_places` (int, default=4): Decimal places

**Output:** Excel file with two sheets:
- `Performance`: Performance metrics table
- `Cost`: Cost metrics table

**Requires:** `openpyxl` package (`pip install openpyxl`)

---

## 1.3 Data Schema

### BaselineRecord

Unified schema for a single benchmark evaluation record.

```python
from baselines import BaselineRecord

record = BaselineRecord(
    dataset_id='aime',
    split='test',
    model_name='gpt-4',
    record_index=0,
    origin_query='What is 2+2?',
    prompt='Q: What is 2+2?\nA:',
    prediction='4',
    raw_output='The answer is 4.',
    ground_truth='4',
    score=1.0,
    prompt_tokens=10,
    completion_tokens=5,
    cost=0.0003
)
```

**Fields:**
- `dataset_id` (str): Dataset identifier
- `split` (str): Dataset split (e.g., 'test', 'train')
- `model_name` (str): Model identifier
- `record_index` (int): Zero-based index within dataset
- `origin_query` (str): Original question from dataset
- `prompt` (str): Formatted prompt sent to model
- `prediction` (str): Extracted answer from model output
- `raw_output` (Any): Complete model response
- `ground_truth` (str): Correct answer
- `score` (float): Evaluation score (typically 0.0 or 1.0)
- `prompt_tokens` (int): Number of tokens in prompt
- `completion_tokens` (int): Number of tokens in completion
- `cost` (float): API cost for this record (USD)

**Methods:**
- `to_dict()`: Convert to dictionary
- `to_dict_compact(include_raw_output=False, include_prompt=True)`: Compact dictionary

---

### AggregatedStats

Summary statistics for a dataset/model combination.

```python
from baselines import AggregatedStats

stats = AggregatedStats(
    dataset_id='aime',
    split='test',
    model_name='gpt-4',
    avg_score=0.85,
    total_records=100,
    correct_records=85,
    total_cost=5.0,
    avg_cost_per_record=0.05,
    total_prompt_tokens=1000,
    total_completion_tokens=500,
    avg_prompt_tokens=10.0,
    avg_completion_tokens=5.0
)
```

**Fields:**
- `dataset_id` (str): Dataset identifier
- `split` (str): Dataset split
- `model_name` (str): Model identifier
- `avg_score` (float): Average score
- `total_records` (int): Total number of records
- `correct_records` (int): Number of correct records
- `total_cost` (float): Total cost
- `avg_cost_per_record` (float): Average cost per record
- `total_prompt_tokens` (int): Total prompt tokens
- `total_completion_tokens` (int): Total completion tokens
- `avg_prompt_tokens` (float): Average prompt tokens
- `avg_completion_tokens` (float): Average completion tokens
- `timestamp` (str, optional): Timestamp of computation

**Properties:**
- `accuracy`: Alias for `avg_score`
- `total_tokens`: Sum of prompt and completion tokens

---

# Part 2: Routing Algorithms & Adaptors

The **Adaptor** module is a unified system that converts LLMRouterBench's standardized data format into algorithm-specific inputs for various LLM routing methods. Each adaptor handles the data transformation, train/test splitting, and format conversion needed for its corresponding routing algorithm.

> **Note**: We recommend creating a separate virtual environment for each algorithm to avoid dependency conflicts.

---

## 2.1 Supported Algorithms

| # | Algorithm | Adaptor |
|:---|:---|:---|
| 1 | **RouterDC** | `routerdc_adaptor` |
| 2 | **EmbedLLM** | `embedllm_adaptor` |
| 3 | **MODEL-SAT** | `modelsat_adaptor` |
| 4 | **GraphRouter** | `graphrouter_adaptor` |
| 5 | **Avengers-Pro** | `avengerspro_adaptor` |
| 6 | **HybridLLM** | `hybridllm_adaptor` |
| 7 | **FrugalGPT** | `frugalgpt_adaptor` |
| 8 | **RouteLLM** | `routellm_adaptor` |

---

## 2.2 Algorithm Quick Start

### RouterDC

```bash
# Generate data
python -m baselines.adaptors.routerdc_adaptor \
    --config config/baseline_config.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --n-clusters 8 \
    --output-dir baselines/RouterDC/data

# Train
cd baselines/RouterDC/train_scripts && ./router_train_7b_seed_42.sh
```

### EmbedLLM

```bash
# Generate data
python -m baselines.adaptors.embedllm_adaptor \
    --config config/baseline_config.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/EmbedLLM/data/small_models

# Train
cd baselines/EmbedLLM/algorithm
export CUDA_VISIBLE_DEVICES=0; python mf.py \
    --train-data-path ../data/small_models/seed42_split0.7/train_ours.csv \
    --test-data-path ../data/small_models/seed42_split0.7/test_ours.csv \
    --question-embedding-path ../data/small_models/seed42_split0.7/question_embeddings.pth \
    --eval-mode router
```

### MODEL-SAT

```bash
# Generate pre-split data
python -m baselines.adaptors.modelsat_adaptor \
    --config config/baseline_config.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/MODEL-SAT/original_data

# Run routing algorithm
cd baselines/MODEL-SAT
# Installation
conda create -n model_sat python=3.10
pip install -r requirements.txt
# Construct routing datasets
./scripts/construct_dataset.sh
# Generate model descriptions
./scripts/generate_model_description.sh
# Routing training
./scripts/train.sh
```

### GraphRouter

```bash
# Generate data
python -m baselines.adaptors.graphrouter_adaptor \
    --baseline-config config/baseline_config_performance_cost.yaml \
    --graphrouter-config baselines/GraphRouter/configs/adaptor_config_seed_42.yaml \
    --embedding-config config/embedding_config.yaml \
    --output-dir baselines/GraphRouter/data/seed_42_0.7 \
    --seed 42

# Train
cd baselines/GraphRouter
export CUDA_VISIBLE_DEVICES=0; python run_exp.py --config configs/config_seed_42_PF.yaml
```

### HybridLLM

```bash
# Generate data (requires exactly 2 models)
python -m baselines.adaptors.hybridllm_adaptor \
    --models "qwen3-235b-a22b-2507,gpt-5" \
    --config config/baseline_config_performance_cost.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/Best-route-llm/data

# Train
cd baselines/Best-route-llm
deepspeed --num_gpus=8 \
    train_router_gte.py \
    --train_data_path data/hybridllm_seed42_split0.7/train.jsonl \
    --test_data_path data/hybridllm_seed42_split0.7/test.jsonl \
    --eval_data_path data/hybridllm_seed42_split0.7/test.jsonl \
    --do_eval True \
    --evaluation_strategy steps \
    --eval_steps 5 \
    --save_strategy steps \
    --save_steps 50 \
    --candidate_models qwen3-235b-a22b-2507,gpt-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --fp16 False \
    --deepspeed config/ds_zero2.json \
    --output_dir outputs/seed42_split0.7 \
    --run_name seed42_split0.7 \
    --num_train_epochs 5
```

### FrugalGPT

```bash
# Generate pre-split data
python -m baselines.adaptors.frugalgpt_adaptor \
    --models "qwen3-235b-a22b-2507,gpt-5" \
    --config config/baseline_config_performance_cost.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/FrugalGPT/original_data

# Run routing algorithm
cd baselines/FrugalGPT
# Installation
conda create -n FrugalGPT python=3.10
pip install git+https://github.com/stanford-futuredata/FrugalGPT
pip install deepspeed
# Routing training
./run_train_local_scorer.sh
```

### Avengers

```bash
# Generate data
python -m baselines.adaptors.avengerspro_adaptor \
    --config config/baseline_config.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/AvengersPro/data/small_models_seed_42

# Run routing
python -m baselines.AvengersPro.simple_cluster_router \
    --config baselines/AvengersPro/config/simple_config_small_models_42.json \
    --output baselines/AvengersPro/logs/simple_config_small_models_42.json
```

### AvengersPro

```bash
# Generate data
python -m baselines.adaptors.avengerspro_adaptor \
    --config config/baseline_config_performance_cost.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/AvengersPro/data/proprietary_models_seed_42

# Run routing
python -m baselines.AvengersPro.simple_cluster_router \
    --config baselines/AvengersPro/config/simple_config_proprietary_models_42.json \
    --output baselines/AvengersPro/logs/simple_config_proprietary_models_42.json

# Run ablation (optional)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_INIT_AT_FORK=FALSE
python -m baselines.AvengersPro.ablation.run_weight_ablation \
    --config baselines/AvengersPro/config/ablation_weight_config_proprietary_models_42.json \
    --output baselines/AvengersPro/ablation/seed_42 \
    --parallel \
    --workers 128
```

### RouteLLM

```bash
# Generate data
python -m baselines.adaptors.routellm_adaptor \
    --config config/baseline_config.yaml \
    --strong-model gpt-5 \
    --weak-model gemini-2.5-flash \
    --output-dir baselines/RouteLLM/data

# Train
python -m baselines.RouteLLM.routers.matrix_factorization.train_matrix_factorization \
    --config baselines/RouteLLM/mf_train_config.json

# Evaluate
python -m baselines.RouteLLM.evaluate_mf \
    --config baselines/RouteLLM/router_eval_config.json \
    --data-dir baselines/RouteLLM/data/seed42_split0.8_gpt-5__vs__gemini-2.5-flash \
    --strong-model gpt-5 \
    --weak-model gemini-2.5-flash \
    --threshold 0.5
```

---

## 2.3 Integrating New Algorithms

### Implementation Steps

1. **Create** adaptor file: `baselines/adaptors/youralgorithm_adaptor.py`
2. **Load** data via `BaselineDataLoader` with unified schema
3. **Split** train/test using `split_by_dataset_then_prompt()` (prevents data leakage)
4. **Transform** data to algorithm-specific format
5. **Organize** algorithm code in `baselines/YourAlgorithm/`

### Core Code Pattern

```python
from collections import defaultdict
from baselines.data_loader import BaselineDataLoader
from baselines.adaptors.common import get_unique_models, fill_missing_models_scores

class YourAlgorithmAdaptor:
    def __init__(self, config_path, random_seed=42, train_ratio=0.8, ood_datasets=None):
        self.loader = BaselineDataLoader(config_path)
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.ood_datasets = ood_datasets or []

    def convert(self, output_dir):
        # 1. Load all records
        all_records = self.loader.load_all_records()

        # 2. Split by dataset then prompt (ensures no data leakage)
        train_records, test_records = self.loader.split_by_dataset_then_prompt(
            all_records, self.train_ratio, self.random_seed, self.ood_datasets
        )

        # 3. Get unique models for consistent feature vectors
        all_models = get_unique_models(all_records)

        # 4. Group by prompt and build scores dict
        prompt_groups = defaultdict(list)
        for record in train_records:
            prompt_groups[(record.dataset_id, record.prompt)].append(record)

        for (dataset_id, prompt), group in prompt_groups.items():
            scores = {r.model_name: r.score for r in group}
            fill_missing_models_scores(scores, all_models, fill_value=0.0)
            # Transform to your algorithm's format...

        # 5. Write output files
        return {'train': train_file, 'test': test_file}
```

### Standard CLI Arguments

| Argument | Description | Default |
|:---|:---|:---|
| `--config` | Baseline config YAML path | Required |
| `--seed` | Random seed for reproducibility | `42` |
| `--split-ratio` | Train set proportion (0-1) | `0.8` |
| `--ood-datasets` | OOD datasets (comma-separated, test only) | `""` |

### Output Structure

```
baselines/YourAlgorithm/data/seed{seed}_split{ratio}/
├── train.{ext}
└── test.{ext}
```

**Reference**: See `baselines/adaptors/avengerspro_adaptor.py` for a complete implementation example.

---

## 🗂️ File Structure

```
baselines/
├── __init__.py
├── README.md               # This file
├── schema.py               # Data schema definitions
├── data_loader.py          # BaselineDataLoader implementation
├── aggregators.py          # BaselineAggregator implementation
├── adaptors/               # Format converters
│   ├── __init__.py
│   ├── common.py
│   ├── routerdc_adaptor.py
│   ├── embedllm_adaptor.py
│   ├── avengerspro_adaptor.py
│   └── ...
├── RouterDC/               # Algorithm implementations
├── EmbedLLM/
├── AvengersPro/
├── GraphRouter/
├── RouteLLM/
└── ...
```

