# 🔌 Collector

The **Collector** module provides unified API access to LLMs with caching, retries, and cost tracking for benchmark data collection.

---

## 🚀 Quick Start

```bash
# Show configuration and available datasets
python -m data_collector.cli info config/data_collector_example.yaml

# Execute benchmark
python -m data_collector.cli run config/data_collector_example.yaml

# List existing results
python -m data_collector.cli list
```

---

## ⚙️ Configuration

See `config/data_collector_example.yaml` for the fully annotated configuration reference.

### Structure

```yaml
models:
  - name: gpt-4o-mini
    api_model_name: openai/gpt-4o-mini
    base_url: https://openrouter.ai/api/v1
    api_key: OPENROUTER_API_KEY
    temperature: 0.2
    timeout: 600

datasets:
  - dataset_id: mmlu_pro
    splits: ["test"]

run:
  output_dir: ./results
  concurrency: 32
  demo_mode: false

cache:
  enabled: true
  mysql:
    host: localhost
    database: openrouterbench_cache
```

### Key Options

| Section | Field | Description |
|:---|:---|:---|
| **models** | `generator_type` | `direct` (text), `multimodal` (vision), `embedding` |
| **run** | `demo_mode` | Test with limited samples |
| **run** | `concurrency` | Parallel API calls |
| **cache** | `enabled` | MySQL caching for cost reduction, resumable runs, and evaluator iteration |

---

## 📄 Result Format

Results saved to: `results/bench/<dataset>/<split>/<model>/<timestamp>.json`

```json
{
  "performance": 0.85,
  "time_taken": 120.5,
  "prompt_tokens": 50000,
  "completion_tokens": 20000,
  "cost": 0.15,
  "records": [
    {
      "index": 1,
      "origin_query": "What is...",
      "prompt": "Question: What is...",
      "prediction": "The answer is...",
      "ground_truth": "42",
      "score": 1.0,
      "prompt_tokens": 50,
      "completion_tokens": 20,
      "cost": 0.001,
      "raw_output": "..."
    }
  ]
}
```

---

## 🔧 Advanced Usage

### Demo Mode

```yaml
run:
  demo_mode: true
  demo_limit: 10  # Only run first 10 samples
```

### Force Overwrite

```bash
python -m data_collector.cli run config.yaml --force
```

### Multimodal Support

```yaml
models:
  - name: gpt-4-vision
    generator_type: multimodal
    api_model_name: openai/gpt-4-vision-preview
```

### MySQL Caching

#### Why Cache?

Enabling cache provides three key benefits:

1. **Dataset Subset Reuse**: Full dataset runs can reuse cached data from subset runs. For example, if you first run 100 samples in `demo_mode`, running the full dataset later will read those 100 samples directly from cache without re-requesting the API.

2. **Resumable Runs**: If a run fails midway due to network issues, API rate limiting, or other errors, restarting will fetch completed requests from cache and only request the remaining samples.

3. **Evaluator Logic Iteration**: When you need to modify evaluation logic (e.g., change answer extraction regex, adjust scoring criteria), enabling `cache_raw_response: true` saves the complete API response. After modifying the evaluator, you can re-evaluate using cached responses without making new API requests.

#### Configuration Reference

```yaml
cache:
  # Basic settings
  enabled: true                    # Enable/disable caching
  force_override_cache: false      # Skip cache read but still write results (useful for rebuilding cache after prompt changes)

  # MySQL connection
  mysql:
    host: localhost                # Database host
    port: 3306                     # Port
    user: null                     # Username (null to read from MYSQL_USER env var)
    password: null                 # Password (null to read from MYSQL_PASSWORD env var)
    database: avengers_cache       # Database name
    table_name: generator_output_cache  # Table name (auto-created on startup)
    charset: utf8mb4               # Character set
    autocommit: true               # Auto-commit mode
    ttl_seconds: null              # Cache TTL in seconds (null = never expire)

    # Connection pool (enable for high concurrency or unstable connections)
    use_connection_pool: false     # Enable connection pooling
    pool_size: 4                   # Pool size
    max_overflow: 2                # Allowed overflow connections
    pool_timeout: 10               # Connection acquire timeout (seconds)
    pool_recycle: 3600             # Connection recycle time (seconds)

  # Cache key generation (determines which request differences create distinct cache entries)
  key_generator:
    # Chat/Completions default: model/temperature/top_p/messages/reasoning_effort together determine the cache key
    # For Embedding mode, set to: ["model", "input"]
    cached_parameters: ["model", "temperature", "top_p", "messages", "reasoning_effort"]
    hash_algorithm: blake2b        # Hash algorithm: blake2b | sha256 | sha1 | md5
    hash_digest_size: 16           # Digest length (effective for blake2b only)

  # Cache write conditions
  conditions:
    cache_successful_only: true    # Only cache successful responses
    min_completion_tokens: 0       # Minimum completion_tokens threshold
    cache_raw_response: false      # Cache complete API response JSON (required for extract_fields)
    refresh_if_missing_raw_response: false  # Re-fetch from API if cached data is missing raw_response

  # Logging and statistics
  log_level: INFO                  # Log level
  enable_stats: true               # Enable cache statistics
```

#### Key Options Explained

| Field | Description |
|:---|:---|
| `force_override_cache` | When `true`, skip cache reads but still write results. Useful for rebuilding cache after prompt changes. |
| `cached_parameters` | Parameters that determine the cache key. Use default for Chat models; set to `["model", "input"]` for Embedding models. |
| `cache_raw_response` | When `true`, saves complete API response. Enables re-evaluation with modified evaluator logic without new API requests. |
| `refresh_if_missing_raw_response` | Used with `cache_raw_response`. Automatically re-fetches when cached data is missing raw_response. |

---

## 🗂️ Architecture

```
data_collector/
├── cli.py           # CLI interface (run/list/clean/info)
├── config_loader.py # YAML configuration parsing
├── planner.py       # Execution planning
├── runner.py        # Concurrent benchmark execution
└── storage.py       # Result storage management
```

### Data Flow

1. Configuration loaded and validated
2. Run plans generated based on existing results
3. Data availability validated for each dataset/split
4. Concurrent execution of benchmark runs
5. Results stored in JSON format with metrics
