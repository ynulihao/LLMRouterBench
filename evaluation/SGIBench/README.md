# SGI-Bench Evaluation Guide

SGI-Bench (Scientific General Intelligence Benchmark) is a benchmark for evaluating the scientific research capabilities of large language models. This implementation covers 4 subtasks of SGI-Bench, fully aligned with the official evaluation logic.

## Subtask Overview

| Subtask | Samples | Evaluation Method | Key Metrics |
|---------|---------|-------------------|-------------|
| Deep Research | 318 | LLM Judge | Exact Match, Step-Level Accuracy |
| Idea Generation | 315 | 3-Model Voting + Embedding Similarity | 4-Dimension Score |
| Dry Experiment | 271 | Code Execution | PassAll@5/3/1, AET, SER |
| Wet Experiment | 68 | Sequence Alignment | Action Similarity, Parameter Accuracy |

---

## 1. Deep Research

**Task Description:** Multi-hop scientific QA requiring information retrieval and computational reasoning.

### Evaluation Pipeline

```
Model Output → Extract <answer> Tag → AnswerParser Normalization → Exact Match
                                                                   ↓
                                                        LLM Judge Step Correctness
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | `answer == extracted_answer OR answer == parsed_answer` |
| **Step-Level Accuracy (SLA)** | LLM step-by-step correctness rate |

### Key Implementation

- **AnswerParser** (`utils.py:AnswerParser`): Uses `gpt-4.1-mini` for answer normalization
  - Numeric answer example: `"0.25"`
  - Text answer example: `"T cell and B cell"`
  - **Note:** Ground truth is not passed to avoid leakage

### Code Entry

```python
# evaluation/SGIBench/deep_research.py
class DeepResearchEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 2. Idea Generation

**Task Description:** Given research background, generate scientific ideas/hypotheses.

### Evaluation Pipeline

```
Model Output (JSON) → Parse 6 Fields → Objective Score (4 dims) → Subjective Voting (3 models × 2 votes)
                                              ↓                           ↓
                                      Embedding Similarity          Win/Lose/Tie Decision
                                      Graph Similarity                    ↓
                                              ↓                    Win=100, else=0
                                      objective_score                     ↓
                                              └──────→ (obj + subj) / 2 ──→ Final Score
```

### Key Metrics

| Dimension | Objective Score | Subjective Decision |
|-----------|-----------------|---------------------|
| **Effectiveness** | Keyword Embedding Similarity × 100 | votes > 3 → win |
| **Novelty** | 0.5×novelty_sim + 0.5×cutting_edge | votes > 4 → win |
| **Detailedness** | 0.2×completeness + 0.4×(10-repetition) + 0.4×(10-length) | votes > 3 → win |
| **Feasibility** | graph_similarity × 100 | votes > 3 → win |

**Final Score = (Effectiveness + Novelty + Detailedness + Feasibility) / 4**

### Voting Mechanism

- **3 Judge Models** × **2 Votes** (normal + swapped positions) = **6 Votes**
- Win Gate: novelty > 4 votes, others > 3 votes
- Subjective Score: win=100, tie/lose=0

### Code Entry

```python
# evaluation/SGIBench/idea_generation.py
class IdeaGenerationEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 3. Dry Experiment

**Task Description:** Complete Python function code for scientific computing.

### Evaluation Pipeline

```
Model Output → Extract Code → Replace in Main Program → Execute 5 Unit Tests → Output Comparison
                                                               ↓
                                                    Exact Match OR LLM Judge
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **PassAll@5** | All 5 tests passed |
| **PassAll@3** | At least 3 tests passed |
| **PassAll@1** | At least 1 test passed |
| **AET** | Average Execution Time |
| **SER** | Successful Execution Rate (successes/5) |

### Special Requirements

1. **Conda Environment**: Must have `dryexp` environment (can be overridden via `SGI_CONDA_ENV`)
2. **Data Fixtures**: Samples 0200, 0206, 0236 require additional data files

### Environment Setup (Dry Experiment Only)

```bash
# Step 1: Create Conda environment
conda create -n dryexp python=3.10 -y
conda activate dryexp

# Step 2: Install dependencies
pip install -r evaluation/SGIBench/dryexp_requirements.txt

# Step 3: Configure environment variables
export SGI_CONDA_ENV="dryexp"
export SGI_DRY_EXPERIMENT_DATA_DIR="/path/to/data/fixtures"  # Required for samples 0200/0206/0236
```

### Code Entry

```python
# evaluation/SGIBench/dry_experiment.py
class DryExperimentEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 4. Wet Experiment

**Task Description:** Generate pseudocode for laboratory experiment protocols.

### Evaluation Pipeline

```
Model Output → Parse Steps → Action Sequence Alignment → Kendall Tau Distance
                    ↓
            Parameter Mapping → Parameter Accuracy
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Action Sequence Similarity** | Kendall tau sequence similarity |
| **Parameter Accuracy** | Parameter matching accuracy |
| **Final Score** | (Sequence Similarity + Parameter Accuracy) / 2 |

### Code Entry

```python
# evaluation/SGIBench/wet_experiment.py
class WetExperimentEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## Code Structure

```
evaluation/SGIBench/
├── __init__.py              # Export 4 evaluators
├── base.py                  # SGIBenchBaseEvaluator base class
├── prompts.py               # Prompt templates
├── utils.py                 # Utility functions (AnswerParser, graph_similarity, etc.)
├── deep_research.py         # DeepResearchEvaluator
├── idea_generation.py       # IdeaGenerationEvaluator
├── dry_experiment.py        # DryExperimentEvaluator
├── wet_experiment.py        # WetExperimentEvaluator
└── dryexp_requirements.txt  # Dry Experiment Conda environment dependencies (68 packages)

data/sgibench/
├── deep_research/test.jsonl    # 318 samples
├── idea_generation/test.jsonl  # 315 samples
├── dry_experiment/test.jsonl   # 271 samples
└── wet_experiment/test.jsonl   # 68 samples
```

---

## Usage Guide

### Quick Start

```bash
# 1. Edit config file to enable desired subtasks
vim config/data_collector_sgibench.yaml

# 2. Run evaluation
./run_sgibench.sh

# Or specify a custom config file
./run_sgibench.sh config/my_custom.yaml
```

### Environment Variables

#### Required Configuration

```bash
# OpenRouter API (model inference)
export OPENROUTER_API_KEY="your-api-key"

# Grader API (shared credentials for all subtasks requiring LLM Judge)
export GRADER_BASE_URL="https://openrouter.ai/api/v1"
export GRADER_API_KEY="$OPENROUTER_API_KEY"
```

#### Deep Research Configuration

```bash
# Grader model (for LLM Judge and AnswerParser)
# Official uses o4-mini, temperature=0
export SGI_DR_GRADER_MODEL="o4-mini"
```

#### Dry Experiment Configuration

```bash
# Grader model (for LLM Judge to determine output correctness)
# Official uses o4-mini, temperature=0
export SGI_DE_GRADER_MODEL="o4-mini"

# Conda environment (must exist)
export SGI_CONDA_ENV="dryexp"                           # Official default

# Execution timeout
export SGI_EXECUTION_TIMEOUT="300"                      # 5 minutes

# Data fixtures directory (required for samples 0200/0206/0236)
export SGI_DRY_EXPERIMENT_DATA_DIR="/path/to/fixtures"
```

#### Idea Generation Configuration

```bash
# Must configure 3 Judge models (official requirement, temperature=0.1)
export SGI_JUDGE_MODEL_1="gpt-5.1-2025-11-13"           # Judge 1
export SGI_JUDGE_MODEL_2="gemini-3-pro-preview"         # Judge 2
export SGI_JUDGE_MODEL_3="anthropic/claude-sonnet-4.5"  # Judge 3

export SGI_JUDGE_BASE_URL_1="https://openrouter.ai/api/v1"
export SGI_JUDGE_BASE_URL_2="https://openrouter.ai/api/v1"
export SGI_JUDGE_BASE_URL_3="https://openrouter.ai/api/v1"

export SGI_JUDGE_API_KEY_1="$OPENROUTER_API_KEY"
export SGI_JUDGE_API_KEY_2="$OPENROUTER_API_KEY"
export SGI_JUDGE_API_KEY_3="$OPENROUTER_API_KEY"

# Embedding model (for objective score calculation)
export SGI_EMBEDDING_MODEL="all-MiniLM-L6-v2"           # Official default
```

#### Wet Experiment Configuration

```bash
# Wet Experiment uses rule-based scoring, no LLM Judge required
#
# Note: The official implementation calls AnswerParser in step_1_get_answer.py,
# but does not use the parsed result in step_2_score.py (line 241 is commented out),
# using the raw answer instead. This implementation follows the same behavior.
```

### Running with Config File

```bash
# Edit config file
vim config/data_collector_sgibench.yaml

# Run
python -m data_collector.cli run config/data_collector_sgibench.yaml
```

### Config File Example

```yaml
models:
  - name: gpt-4o
    api_model_name: openai/gpt-4o
    base_url: https://openrouter.ai/api/v1
    api_key: OPENROUTER_API_KEY
    temperature: 0.0
    timeout: 600

datasets:
  - dataset_id: sgibench-deepresearch
    splits: ["test"]
  - dataset_id: sgibench-wetexperiment
    splits: ["test"]
  # - dataset_id: sgibench-ideageneration  # Requires 3 judge models
  # - dataset_id: sgibench-dryexperiment   # Requires dryexp conda environment

run:
  output_dir: ./results
  concurrency: 8
  demo_mode: true
  demo_limit: 10
```

---

## Official Alignment Notes

This implementation is fully aligned with the official SGI-Bench evaluation logic:

### Grader Configuration Alignment

| Subtask | Official Grader Model | Temperature | Environment Variable |
|---------|----------------------|-------------|---------------------|
| Deep Research | `o4-mini` | 0 | `SGI_DR_GRADER_MODEL` |
| Dry Experiment | `o4-mini` | 0 | `SGI_DE_GRADER_MODEL` |
| Idea Generation | 3 Judge Models | 0.1 | `SGI_JUDGE_MODEL_1/2/3` |
| Wet Experiment | No LLM Judge | - | - |

### Evaluation Logic Alignment

| Alignment Item | Official Implementation | Local Implementation |
|----------------|------------------------|----------------------|
| DR/DE Grader | `o4-mini`, temp=0 | Aligned |
| IG Judge Models | `["gpt-5.1-2025-11-13", "gemini-3-pro-preview", "anthropic/claude-sonnet-4.5"]` | Aligned |
| IG Temperature | 0.1 | Aligned |
| AnswerParser Examples | `"0.25"` / `"T cell and B cell"` | Aligned |
| Voting Win Gate | novelty>4, others>3 | Aligned |
| Subjective Score | win=100, else=0 | Aligned |
| graph_similarity | Sequential Edge Fallback | Aligned |
| Dry Experiment Env | `dryexp` conda | Enforced |
| DE data_en.py Execution | step_1_build.py pre-runs | Runs Before Each Test |
| DE data_en.py Timeout | 10 minutes | Aligned |

---

## Dependencies

```bash
# Base dependencies (Idea Generation)
pip install sentence-transformers  # Embedding computation
pip install networkx               # Graph similarity
pip install json-repair            # JSON parsing fix (optional)

# Dry Experiment environment
conda create -n dryexp python=3.10 -y
conda activate dryexp
pip install -r evaluation/SGIBench/dryexp_requirements.txt
```

---

## References

- Official Code: `SGI-Bench/evaluation/`
- Paper: SGI-Bench: A Benchmark for Scientific General Intelligence
