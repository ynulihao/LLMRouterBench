<div align="center">

<img src="assets/logo.png" width="160" alt="OpenRouterBench">

# OpenRouterBench

### A One-Stop Benchmark and Solution Suite for LLM Routing

![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/NPULH/OpenRouterBench)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()


</div>

---

<div align="center">
<img src="assets/framework.png" width="95%" alt="Framework">
</div>

<br>

## 🌟 Highlights
**OpenRouterBench provides high-quality inference data collected with nearly $3,000 and hundreds of GPU-hours, including:**

- Covering **25+ diverse and challenging datasets (e.g., HLE, SimpleQA, SWE-Bench)** across multiple domains.
- Including both **20 lightweight ~7B models (e.g., Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B)** and **13 flagship models (e.g., GPT-5, Gemini-2.5-Pro)**.
- Providing **inference costs from OpenRouter** for all flagship models, including both USD and tokens.
- Offering a **unified, plug-and-play framework** for fair cross-method comparisons.
- Supporting **both performance and performance-cost routing paradigms**.
- Releasing **complete data** with per-prompt, per-model predictions and evaluations.

<!-- Convert assets/framework.pdf to assets/framework.png -->
<br>



## 🚀 Installation

```bash
git clone https://github.com/ynulihao/OpenRouterBench.git
cd OpenRouterBench
pip install -r requirements.txt
```

---

## ⚙️ Configurations

OpenRouterBench supports two routing paradigms with corresponding configuration files:

| Setting | Description | Collector Config | Adaptor Config |
|:---|:---|:---|:---|
| **Performance** | 20 lightweight models (~7B) | `config/data_collector_small_model_config.yaml` | `config/baseline_config.yaml` |
| **Performance-Cost** | 13 flagship models with cost | `config/data_collector_proprietary_model_config.yaml` | `config/baseline_config_performance_cost.yaml` |

---

## 🧩 Core Components

OpenRouterBench provides a modular three-component architecture:

<table>
<tr>
<th width="33%">🔌 Collector</th>
<th width="33%">🔍 Evaluator</th>
<th width="33%">🔀 Adaptor</th>
</tr>
<tr>
<td valign="top">

Unified API interface to LLMs:
- Caching & retries
- Cost tracking
- Token counting

[Documentation →](data_collector/README.md)

</td>
<td valign="top">

Dataset-specific evaluation for 25+ benchmarks:
- Math: AIME, MATH500
- Code: HumanEval, MBPP
- Knowledge: MMLU-Pro, GPQA

[Documentation →](evaluation/README.md)

</td>
<td valign="top">

Format conversion for 8 routing algorithms:

RouterDC, EmbedLLM, GraphRouter, Avengers-Pro, HybridLLM, FrugalGPT, RouteLLM, MODEL-SAT

[Documentation →](baselines/README.md)

</td>
</tr>
</table>

---

## 🎯 Quick Start

<table>
<tr>
<td>

**1. Collect Data**
```bash
python -m data_collector.cli run \
    config/baseline_config.yaml
```

</td>
<td>

**2. Analyze**
```python
from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader("config/baseline_config.yaml")
records = loader.load_all_records()
agg = BaselineAggregator(records, data_loader=loader)
agg.print_summary_tables(
    score_as_percent=True,
    test_mode=False
)
```

</td>
<td>

**3. Train Router**
```bash
python -m baselines.adaptors.avengerspro_adaptor \
    --config config/baseline_config.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/AvengersPro/data/small_models_seed_42

python -m baselines.AvengersPro.simple_cluster_router \
    --config baselines/AvengersPro/config/simple_config_small_models_42.json \
    --output baselines/AvengersPro/logs/simple_config_small_models_42.json
```

</td>
</tr>
</table>

---

## 📄 Data Format

<details>
<summary><b>Result File Structure</b></summary>

Results are stored in JSON format at `results/bench/<dataset>/<split>/<model>/<timestamp>.json`:

```json
{
  "performance": 0.85,
  "time_taken": 120.5,
  "prompt_tokens": 50000,
  "completion_tokens": 20000,
  "cost": 0.15,
  "counts": 100,
  "records": [
    {
      "index": 1,
      "origin_query": "What is the sum of 2+2?",
      "prompt": "Question: What is the sum of 2+2?\nAnswer:",
      "prediction": "4",
      "ground_truth": "4",
      "score": 1.0,
      "prompt_tokens": 15,
      "completion_tokens": 5,
      "cost": 0.0001,
      "raw_output": "The sum of 2+2 is 4."
    }
  ]
}
```

</details>

<details>
<summary><b>Data Download</b></summary>

Download pre-collected benchmark results:

**Baidu Netdisk：** [bench-release.tar.gz](https://pan.baidu.com/s/1bfa_eX3bhuo7wgNlD_dbpA?pwd=mmbf) (code：`mmbf`)

**Google Drive：** [bench-release.tar.gz](https://drive.google.com/file/d/12pupoZDjqziZ2JPspH60MCC8fdXWgnX1/view?usp=drive_link)

**Hugging Face：** [bench-release.tar.gz](https://huggingface.co/datasets/NPULH/OpenRouterBench)

```bash
# Extract to results directory
tar xzf bench.tar.gz
```

Directory structure after extraction:
```
results/
└── bench/
    ├── aime/
    ├── bbh/
    ├── humaneval/
    ├── mmlu_pro/
    └── ...
```

See [results/download.md](results/download.md) for details.

</details>

<details>
<summary><b>Data Viewer Example</b></summary>

```python
from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader('config/baseline_config.yaml')
records = loader.load_all_records()

test_mode = False
score_as_percent = True
train_ratio = 0.7
random_seed = 3407

agg = BaselineAggregator(records, data_loader=loader)
agg.print_summary_tables(
    score_as_percent=score_as_percent,
    test_mode=test_mode,
    random_seed=random_seed,
    train_ratio=train_ratio,
)

agg.save_summary_tables_to_excel(
    output_file='small_models_total.xlsx',
    score_as_percent=score_as_percent,
    test_mode=test_mode,
    train_ratio=train_ratio,
    random_seed=random_seed
)
```

</details>

---

## 📊 Datasets

<details>
<summary><b>Performance Setting (15 Datasets)</b></summary>

| Category | Dataset | Abbrev. | Samples | Metric |
|:---|:---|:---:|---:|:---:|
| **Math** | AIME | AIME | 60 | Accuracy, 0-shot |
| | MATH500 | M500. | 500 | Accuracy, 0-shot |
| | MATHBench | MBen. | 150 | Accuracy, 0-shot |
| **Code** | HumanEval | HE. | 164 | Pass@1, 0-shot |
| | MBPP | MBPP | 974 | Pass@1, 0-shot |
| | LiveCodeBench | LCB. | 1055 | Pass@1, 0-shot |
| **Logic** | BBH | BBH | 1080 | Accuracy, 3-shot |
| | KORBench | KOR. | 1250 | Accuracy, 3-shot |
| | Knights & Knaves | K&K. | 700 | Accuracy, 0-shot |
| **Knowledge** | MMLU-Pro | MP. | 1000 | Accuracy, 0-shot |
| | GPQA | GPQA | 198 | Accuracy, 0-shot |
| | FinQA | FQA. | 1147 | Accuracy, 0-shot |
| | MedQA | MQA. | 1273 | Accuracy, 0-shot |
| **Affective** | EmoryNLP | Emory. | 697 | Accuracy, 0-shot |
| | MELD | MELD | 1232 | Accuracy, 0-shot |

</details>

<details>
<summary><b>Performance-Cost Setting (10 Datasets)</b></summary>

| Category | Dataset | Abbrev. | Samples | Metric |
|:---|:---|:---:|---:|:---:|
| **Math** | AIME | AIME | 60 | Accuracy, 0-shot |
| | LiveMathBench | LMB. | 121 | Accuracy, 0-shot |
| **Code** | LiveCodeBench | LCB. | 1055 | Pass@1, 0-shot |
| | SWE-Bench | SWE. | 500 | Pass@1, 0-shot |
| **Knowledge** | GPQA | GPQA | 198 | Accuracy, 0-shot |
| | HLE | HLE | 2158 | LLM as judge, 0-shot |
| | MMLU-Pro | MP. | 3000 | Accuracy, 0-shot |
| | SimpleQA | SQA. | 4326 | LLM as judge, 0-shot |
| **Instruction** | ArenaHard | AHARD. | 750 | LLM as judge, 0-shot |
| **Agentic** | τ²-Bench | TAU2. | 278 | Success Rate, 0-shot |

</details>

---

## 🤖 Model Pools

<details>
<summary><b>Performance Setting (20 Models)</b></summary>

| Model | Abbr. | Params |
|:---|:---:|:---:|
| DeepHermes-3-Llama-3-8B-Preview | DH-Llama3-it | 8B |
| DeepSeek-R1-0528-Qwen3-8B | DS-Qwen3 | 8B |
| DeepSeek-R1-Distill-Qwen-7B | DS-Qwen | 7B |
| Fin-R1 | Fin-R1 | 7B |
| GLM-Z1-9B-0414 | GLM-Z1 | 9B |
| Intern-S1-mini | Intern-S1-mini | 8B |
| Llama-3.1-8B-Instruct | Llama-3.1-it | 8B |
| Llama-3.1-8B-UltraMedical | UltraMedical | 8B |
| Llama-3.1-Nemotron-Nano-8B-v1 | Llama-Nemo | 8B |
| MiMo-7B-RL-0530 | MiMo-RL | 7B |
| MiniCPM4.1-8B | MiniCPM | 8B |
| NVIDIA-Nemotron-Nano-9B-v2 | NVIDIA-Nemo | 9B |
| OpenThinker3-7B | OpenThinker | 7B |
| Qwen2.5-Coder-7B-Instruct | Qwen-Coder | 7B |
| Qwen3-8B | Qwen3-8B | 8B |
| Cogito-v1-preview-llama-8B | Cogito-v1 | 8B |
| Gemma-2-9b-it | Gemma-2-it | 9B |
| Glm-4-9b-chat | Glm-4-chat | 9B |
| Granite-3.3-8b-instruct | Granite-3.3-it | 8B |
| Internlm3-8b-instruct | Internlm3-it | 8B |

</details>

<details>
<summary><b>Performance-Cost Setting (13 Models)</b></summary>

| Model | Abbr. | Input Price | Output Price |
|:---|:---:|---:|---:|
| Claude-sonnet-4 | Claude-v4 | $3.00/1M | $15.00/1M |
| Gemini-2.5-flash | Gemini-Flash | $0.30/1M | $2.50/1M |
| Gemini-2.5-pro | Gemini-Pro | $1.25/1M | $10.00/1M |
| GPT-5-chat | GPT-5-Chat | $1.25/1M | $10.00/1M |
| GPT-5-medium | GPT-5 | $1.25/1M | $10.00/1M |
| Qwen3-235b-a22b-2507 | Qwen3-235B | $0.09/1M | $0.60/1M |
| Qwen3-235b-a22b-thinking-2507 | Qwen3-Thinking | $0.30/1M | $2.90/1M |
| Deepseek-v3-0324 | DeepSeek-V3 | $0.25/1M | $0.88/1M |
| Deepseek-v3.1-terminus | DS-V3.1-Tms | $0.27/1M | $1.00/1M |
| Deepseek-r1-0528 | DeepSeek-R1 | $0.50/1M | $2.15/1M |
| GLM-4.6 | GLM-4.6 | $0.60/1M | $2.20/1M |
| Kimi-k2-0905 | Kimi-K2 | $0.50/1M | $2.00/1M |
| Intern-s1 | Intern-S1 | $0.18/1M | $0.54/1M |

</details>

---

## 📈 Model Performance on Datasets

### Routing for Performance Setting

Performance of 20 lightweight ~7B models across 15 datasets spanning Mathematics, Code, Logic, Knowledge, and Affective domains. Deep red and light red highlight best and second-best results.

<div align="center">
<img src="assets/Table6.png" width="95%" alt="Table 6: Model Performance - Performance Setting">
</div>

### Routing for Performance-Cost Tradeoff Setting

Performance of 13 flagship models on 10 datasets covering Mathematics, Code, Knowledge, Instruction Following, and Agentic tasks.

<div align="center">
<img src="assets/Table9.png" width="95%" alt="Table 9: Model Performance - Performance-Cost Setting">
</div>

<br>

Inference cost comparison across models and datasets ($/1M tokens). Shows the cost heterogeneity that enables cost-aware routing.

<div align="center">
<img src="assets/Table10.png" width="95%" alt="Table 10: Model Inference Costs">
</div>

---

## 🔬 Experimental Results

### Performance-Oriented Routing

The table below shows routing performance metrics across different methods:

- **Acc**: Overall accuracy averaged across all datasets.
- **ImpRand**: an improvement over Random Router, a baseline that uniformly selects models.
- **ImpMax**: an improvement over Max Expert, a hindsight baseline that, for each dataset, selects the single model achieving the highest accuracy on that dataset and then uses only this model for all instances in that dataset.
- **OracleGap**: gap to Oracle, an instance-level upper bound that always picks the best available model for each query.

**Key Findings**: Routers nearly match the Max Expert baseline but remain far below the Oracle, indicating substantial room for improvement in instance-level routing.

<div align="center">
<img src="assets/Table7.png" width="70%" alt="Table 7: Performance Metrics">
</div>

### Cost-Aware Routing

Performance-cost tradeoff metrics for routing methods:

- **PerfGain**: Maximum performance improvement over the best single model.
- **CostSave**: Cost reduction achieved while maintaining comparable accuracy to the best single model.
- **ParetoRatio**: How often a method dominates competing methods (higher is better).
- **ParetoDist**: Average distance to the Pareto frontier (lower is better).

**Key Findings**: Routing methods achieve comparable accuracy to top models at significantly reduced inference costs, yet gaps to Max Expert and Oracle remain.

<div align="center">
<img src="assets/Table8.png" width="70%" alt="Table 8: Cost-Aware Routing Results">
</div>

### Performance-Cost Tradeoffs

Visualization of accuracy vs. cost across all routing methods and base models. Panel A shows the Pareto frontier, while Panel B quantifies performance gains and cost savings relative to best single model (GPT-5).

<div align="center">
<img src="assets/Figure3.png" width="95%" alt="Figure 3: Performance-Cost Tradeoffs">
</div>

---

## 🗂️ Project Structure

```
OpenRouterBench/
├── data_collector/     # Collector module
├── evaluation/         # Evaluator (25+ datasets)
├── baselines/          # Adaptor & routing algorithms
├── generators/         # Model API interface
├── common/cache/       # Caching system
├── external_bench/     # Third-party integration
├── config/             # Configuration files
└── results/            # Benchmark results
```

## 🗓️ Roadmap

OpenRouterBench is evolving to better support the research community:

- **Research Publication**: An academic paper describing OpenRouterBench will be released.
- **Expanded Model Evaluations**: Additional benchmarks of flagship models will be provided.
- **Broader Dataset Coverage**: More datasets across diverse domains will be integrated.

---

## 📚 Related Work

### Comparison with Existing Routing Benchmarks

<div align="center">
<img src="assets/Table1_2.png" width="95%" alt="Comparison with Existing Routing Benchmarks">
</div>

<br>

Existing routing benchmarks face several limitations:

- **RouterBench**: Restricted to early-generation models and 8 relatively simple datasets.
- **EmbedLLM & RouterEval**: Focus on open-source models without inference cost information.
- **FusionFactory**: Benchmarks open-source models with estimated costs.
- **RouterArena**: Uses inconsistent model pools across routers, undermining fair comparison and lacking per-prompt, per-model data.

## 📝 Citation

If you find OpenRouterBench useful, please cite our paper:
```bibtex
@article{openrouterbench,
  title={OpenRouterBench: A Massive Benchmark for LLM Routing},
  author={Li, Hao and Zhang, Yiqun and Wang, Chenxu and Guo, Zhaoyan and Chen, Jianhao and Zhang, Hangfan and Tang, Shengji and Zhang, Qiaosheng and Ye, Peng and Chen, Yang and Bai, Lei and Wang, Zhen and Hu, Shuyue},
  note={Coming soon},
  year={2025}
}
```
This work is part of our series of studies on LLM routing; if you’re interested, please refer to and cite:
```bibtex
@article{zhang2025avengers,
  title={The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants},
  author={Zhang, Yiqun and Li, Hao and Wang, Chenxu and Chen, Linyao and Zhang, Qiaosheng and Ye, Peng and Feng, Shi and Wang, Daling and Wang, Zhen and Wang, Xinrun and others},
  journal={arXiv preprint arXiv:2505.19797},
  year={2025}
}
@article{zhang2025beyond,
  title={Beyond gpt-5: Making llms cheaper and better via performance-efficiency optimized routing},
  author={Zhang, Yiqun and Li, Hao and Chen, Jianhao and Zhang, Hangfan and Ye, Peng and Bai, Lei and Hu, Shuyue},
  journal={arXiv preprint arXiv:2508.12631},
  year={2025}
}
@article{wang2025icl,
  title={ICL-Router: In-Context Learned Model Representations for LLM Routing},
  author={Wang, Chenxu and Li, Hao and Zhang, Yiqun and Chen, Linyao and Chen, Jianhao and Jian, Ping and Ye, Peng and Zhang, Qiaosheng and Hu, Shuyue},
  journal={arXiv preprint arXiv:2510.09719},
  year={2025}
}
@article{chen2025learning,
  title={Learning Compact Representations of LLM Abilities via Item Response Theory},
  author={Chen, Jianhao and Wang, Chenxu and Zhang, Gengrui and Ye, Peng and Bai, Lei and Hu, Wei and Qu, Yuzhong and Hu, Shuyue},
  journal={arXiv preprint arXiv:2510.00844},
  year={2025}
}
```

---

<div align="center">

**OpenRouterBench** — Advancing LLM Routing Research

[Report Issue](https://github.com/ynulihao/OpenRouterBench/issues) · [Request Feature](https://github.com/ynulihao/OpenRouterBench/issues)

</div>
