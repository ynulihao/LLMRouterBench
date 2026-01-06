# 🔧 Tools

The **Tools** module provides utility scripts for LLMRouterBench system maintenance, result management, and debugging.

---

## 📑 Tool List

| # | Tool | Purpose |
|:---|:---|:---|
| 1 | `cache_writer` | Write benchmark results into MySQL cache |
| 2 | `visualize_embeddings` | Visualize question embeddings with dimensionality reduction |
| 3 | `test_embedding_model` | Sanity-check embedding model configuration |

---

## 1️⃣ cache_writer

Write existing benchmark results from `results/bench` into the MySQL cache to avoid repeated API costs during development.

### 🎯 Purpose

If you have already run benchmarks and want to avoid re-running the same API calls during development or testing, this tool writes the existing results into the cache system.

### ✨ Features

- Scans result files in `results/bench` directory
- Supports filtering by datasets and models
- Extracts model names and normalizes cache keys
- Batch processing with progress tracking
- Configurable retry logic with exponential backoff
- Dry-run mode for testing
- Write verification to ensure cache integrity

### 📖 Usage

```bash
# Basic usage
python -m tools.cache_writer config/cache_write_config.yaml

# Test mode (shows what would be written without actually writing)
python -m tools.cache_writer config/cache_write_config.yaml --dry-run

# Verbose logging
python -m tools.cache_writer config/cache_write_config.yaml --verbose

# Filter specific models
python -m tools.cache_writer config/cache_write_config.yaml \
    --include-models gpt-4,claude-3 \
    --exclude-models failed-model
```

### ⚙️ Configuration

Example configuration file (`config/cache_write_config.yaml`):

```yaml
cache_writer:
  results_directory: "results/bench"
  skip_benchmarks: []              # Benchmarks to skip
  include_datasets: []             # Empty = all datasets
  temperature: 0.2                 # Temperature for cache key
  top_p: 1.0                       # Top_p for cache key
  batch_size: 1024                 # Batch size for processing
  show_progress: true              # Enable progress reporting
  max_records: null                # Maximum records (null = all)
  max_retries: 10                  # Maximum retry attempts
  retry_delay: 0.1                 # Base delay between retries
  enable_write_verification: true  # Verify cache writes
  include_models: []               # Models to include (empty = all)
  exclude_models: []               # Models to exclude

cache:
  enabled: true
  mysql:
    host: your-mysql-host
    port: 3306
    user: username
    password: password
    database: cache_db
    table_name: generator_output_cache
  key_generator:
    cached_parameters: ["model", "temperature", "top_p", "messages", "reasoning_effort"]
```

### 📋 Configuration Fields

| Field | Type | Required | Default | Description |
|:---|:---|:---:|:---:|:---|
| `cache_writer.results_directory` | string | ✓ | - | Directory containing result files |
| `cache_writer.skip_benchmarks` | list[string] | ✗ | [] | Benchmarks to skip |
| `cache_writer.include_datasets` | list[string] | ✗ | [] | Datasets to include (empty = all) |
| `cache_writer.temperature` | float | ✗ | 0.2 | Temperature for cache key |
| `cache_writer.top_p` | float | ✗ | 1.0 | Top_p for cache key |
| `cache_writer.batch_size` | int | ✗ | 1024 | Batch processing size |
| `cache_writer.show_progress` | bool | ✗ | true | Enable progress reporting |
| `cache_writer.max_records` | int | ✗ | null | Maximum records (null = all) |
| `cache_writer.max_retries` | int | ✗ | 10 | Maximum retry attempts |
| `cache_writer.retry_delay` | float | ✗ | 0.1 | Base retry delay (exponential backoff) |
| `cache_writer.enable_write_verification` | bool | ✗ | true | Verify writes by reading back |
| `cache_writer.include_models` | list[string] | ✗ | [] | Models to include (empty = all) |
| `cache_writer.exclude_models` | list[string] | ✗ | [] | Models to exclude |

### 📄 Output Example

```
2025-10-XX XX:XX:XX | INFO     | Starting cache writer...
2025-10-XX XX:XX:XX | INFO     | Results directory: results/bench
2025-10-XX XX:XX:XX | INFO     | Found 50 result files to process
2025-10-XX XX:XX:XX | INFO     | Processing file 1/50: aime-2024-gpt-4.json
...
=== CACHE WRITER STATISTICS ===
Files processed: 50
Files skipped: 0
Records processed: 25000
Records cached: 24850
Records failed: 150
Success rate: 99.4%
```

### 📝 Notes

- Script automatically creates database and table if they don't exist
- Recommended to use `--dry-run` mode for first-time testing
- For large datasets, use `max_records` to limit test batches
- All errors and warnings are logged

---

## 2️⃣ visualize_embeddings

Visualize question embeddings from benchmark results using dimensionality reduction, colored by dataset.

### 🎯 Purpose

Generate 2D visualizations of question embeddings to understand dataset clustering patterns and similarity structures across different benchmarks.

### ✨ Features

- Supports multiple dimensionality reduction methods: t-SNE, PCA, UMAP
- Colors points by dataset for easy comparison
- Sample limiting to handle large datasets
- Configurable embedding model
- Generates publication-ready PNG visualizations

### 📖 Usage

```bash
# Default visualization (t-SNE, all samples)
python -m tools.visualize_embeddings

# Limit samples per dataset
python -m tools.visualize_embeddings --max-samples 100

# Use different dimensionality reduction method
python -m tools.visualize_embeddings --method pca
python -m tools.visualize_embeddings --method umap

# Custom configuration and output
python -m tools.visualize_embeddings \
    --config config/embedding_config.yaml \
    --output embedding_viz.png \
    --method tsne \
    --max-samples 200

# With verbose logging
python -m tools.visualize_embeddings --verbose
```

### 📋 Options

| Option | Description | Default |
|:---|:---|:---|
| `--config` | Path to embedding configuration YAML | `config/embedding_config.yaml` |
| `--method` | Dimensionality reduction: `tsne`, `pca`, `umap` | `tsne` |
| `--max-samples` | Maximum samples per dataset | all |
| `--output` | Output image path | `embedding_visualization_<model>.png` |
| `--verbose` | Enable verbose logging | false |

### ⚙️ Configuration

Example embedding configuration (`config/embedding_config.yaml`):

```yaml
embedding_model:
  name: gte_Qwen2-7B-instruct
  api_model_name: gte_Qwen2-7B-instruct
  generator_type: embedding
  base_url: https://your-embedding-api-url/v1
  api_key: HUOSHAN_API_KEY  # Environment variable name
  timeout: 600

# Optional cache configuration
cache:
  enabled: true
  mysql:
    host: your-mysql-host
    port: 3306
    user: username
    password: password
    database: cache_db
    table_name: generator_output_cache
```

### 📋 Configuration Fields

| Field | Type | Required | Default | Description |
|:---|:---|:---:|:---:|:---|
| `embedding_model.name` | string | ✓ | - | Embedding model identifier |
| `embedding_model.api_model_name` | string | ✓ | - | API model name |
| `embedding_model.generator_type` | string | ✓ | `"embedding"` | Must be `"embedding"` |
| `embedding_model.base_url` | string | ✓ | - | Embedding API base URL |
| `embedding_model.api_key` | string | ✓ | - | API key or env var name |
| `embedding_model.timeout` | int | ✗ | 600 | Request timeout (seconds) |

### 📦 Dependencies

```bash
# Core dependencies (always required)
pip install numpy matplotlib scikit-learn

# For UMAP method
pip install umap-learn
```

### 📄 Output

Generates a scatter plot PNG image with:
- Each point represents a question
- Colors indicate dataset membership
- Shows embedding space structure and dataset clustering
- Legend with all datasets

### 📝 Notes

- Large datasets may take significant time to process
- Use `--max-samples` to reduce processing time
- UMAP generally produces better clustering than t-SNE or PCA
- Enable caching in config to avoid re-computing embeddings

---

## 3️⃣ test_embedding_model

Sanity-check the configured embedding model by verifying API connectivity and response format.

### 🎯 Purpose

Quickly verify that the embedding model configuration is correct and the API returns valid embeddings with consistent dimensionality before running full-scale operations.

### ✨ Features

- Test single prompts or batch from file
- Validates embedding vector dimensionality consistency
- Reports token usage for each embedding
- Supports all embedding configurations used by other tools

### 📖 Usage

```bash
# Test with a single prompt
export HUOSHAN_API_KEY=your-api-key
python -m tools.test_embedding_model --text "Hello LLMRouterBench!"

# Test with prompts from a file (one per line)
python -m tools.test_embedding_model --file prompts.txt

# Use custom configuration
python -m tools.test_embedding_model \
    --config config/embedding_config.yaml \
    --text "Test embedding generation"
```

### 📋 Options

| Option | Description | Default |
|:---|:---|:---|
| `--config` | Path to embedding configuration YAML | `config/embedding_config.yaml` |
| `--text` | Single prompt to embed | - |
| `--file` | File containing prompts (one per line) | - |

### ⚙️ Configuration

Uses the same embedding configuration as `visualize_embeddings`:

```yaml
embedding_model:
  name: gte_Qwen2-7B-instruct
  api_model_name: gte_Qwen2-7B-instruct
  generator_type: embedding
  base_url: https://your-embedding-api-url/v1
  api_key: HUOSHAN_API_KEY  # Environment variable name
  timeout: 600
```

### 📄 Output Example

```
2025-10-XX XX:XX:XX | INFO     | Initialised embedding generator for model gte_Qwen2-7B-instruct
2025-10-XX XX:XX:XX | INFO     | Prompt: Hello LLMRouterBench! | dim=3584 | prompt_tokens=5
2025-10-XX XX:XX:XX | INFO     | Embedding test complete: 1 succeeded, 0 failed
```

### 📝 Notes

- Requires `--text` or `--file` argument
- Environment variable for API key must be set before running
- Returns exit code 0 on success, 1 on failure

---

## 🗂️ File Structure

```
tools/
├── __init__.py
├── README.md                  # This file
├── cache_writer.py            # Cache management tool
├── visualize_embeddings.py    # Embedding visualization tool
└── test_embedding_model.py    # Embedding model testing tool
```

---

## 📊 Quick Reference

| Tool | Purpose | Input | Output |
|:---|:---|:---|:---|
| `cache_writer` | Cache benchmark results | `results/bench/` | MySQL cache |
| `visualize_embeddings` | Visualize embeddings | Benchmark questions | PNG visualization |
| `test_embedding_model` | Test embedding API | Text prompt(s) | Console output |
