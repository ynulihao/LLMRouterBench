<div align="center">

<img src="assets/logo.png" width="160" alt="LLMRouterBench">

# LLMRouterBench

### A Massive Benchmark and Unified Framework for LLM Routing

[English](README.md) | [中文](README_zh.md)

![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/NPULH/LLMRouterBench)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#model-pools">Model Pools</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

---

## Overview

<div align="center">

**33 models | 21+ datasets | 10 routing algorithms | 400K+ instances | ~1.8B tokens**

</div>

LLMRouterBench is a large-scale benchmark and unified framework for LLM routing. It consolidates standardized outputs across 21 datasets and 33 models, supports both performance-only and performance-cost routing, and provides adapters for 10 representative routing baselines.

<div align="center">
<img src="assets/PPT_fig1.png" width="95%" alt="LLMRouterBench Overview">
</div>

### What's Included

- **Two Routing Paradigms**: **Performance-oriented**  | **Performance-Cost** tradeoff
- **Modular Architecture**: **Collector** (unified LLM API) → **Evaluator** (dataset-specific scoring) → **Adaptor** (algorithm-specific formatting)
- **State-of-the-Art Model Pools**: 20 **lightweight** ~7B LLMs (Qwen3-8B, DS-Qwen3, NVIDIA-Nemo, etc.) + 13 **flagship** LLMs from 8 providers (GPT-5, Gemini-2.5-Pro, Claude-4, DeepSeek-V3.1, etc.)
- **Challenging & Diverse Datasets**: **Math** (AIME, LiveMathBench), **Code** (LiveCodeBench, SWE-Bench), **Logic** (BBH, KORBench), **Knowledge** (HLE, SimpleQA), **Affective** (EmoryNLP, MELD), **Instruction Following** (ArenaHard), **Tool Use** (τ²-Bench)
- **Representative Routing Baselines**: RouterDC (NeurIPS'24), EmbedLLM (ICLR'25), MODEL-SAT (AAAI'25), Avengers (AAAI'26), HybridLLM (ICLR'24), FrugalGPT (TMLR'24), RouteLLM (ICLR'25), GraphRouter (ICLR'25), **Avengers-Pro** (DAI'25 Best Paper), OpenRouter
- **Substantial Data Collection Cost**: ~1K GPU hours + $3000 API spend

- **Standardized Data Fields** (per instance): `origin_query`, `prompt`, `prediction(raw output)`, `ground_truth`, `score`, `prompt_tokens`, `completion_tokens`, `cost`

---

## Key Findings

### Performance-Oriented Setting

**No single model rules every domain; models exhibit complementary strengths.** As shown below, mathematics benchmarks are led by models like Intern-S1-mini or Qwen3-8B, code benchmarks by Qwen-Coder or Fin-R1, confirming the central premise of LLM routing.

<div align="center">
<img src="assets/Figure8-perf-main-1-row.png" width="95%" alt="Model Performance Across Domains">
</div>

**Top routing methods achieve comparable performance, but a significant gap to Oracle remains.** We compare routing methods against three baselines: **Random** (randomly selects a model), **Best Single** (single model with highest average accuracy), and **Oracle** (always selects the best model per query—theoretical upper bound). Key metrics include:
- **AvgAcc**: Average accuracy across all datasets
- **Gain@R / Gain@B**: Relative improvement over Random / Best Single
- **Gap@O**: Gap to Oracle (lower is better)

Despite methodological differences, leading routers (EmbedLLM, GraphRouter, MODEL-SAT, Avengers) yield similar results. Notably, Avengers achieves this without neural-network training. The proximity to **Dataset Oracle** (which selects the best model per dataset, shown as hatched bars) suggests that routing gains largely stem from capturing coarse-grained domain structure. However, a significant Gap@O remains due to **model-recall failures**—when only a few models answer correctly, routers often fail to select them.

<div align="center">
<img src="assets/Figure7-perf-metrics-03.png" width="95%" alt="Performance Metrics">
</div>

### Performance-Cost Setting

**Effective routing achieves significant gains, but not all routers succeed.** We measure performance-cost tradeoffs using:
- **PerfGain**: Best achievable performance improvement over Best Single (at highest-accuracy configuration)
- **CostSave**: Maximal cost reduction while maintaining Best Single's accuracy
- **Pareto frontier**: The set of optimal configurations where no method is simultaneously cheaper and more accurate
- **ParetoDist**: Average distance to Pareto frontier (smaller is better)

Top methods achieve up to 4% PerfGain and 31.7% CostSave. However, some routers (including commercial ones like OpenRouter) fail to outperform the Best Single. Avengers-Pro dominates the Pareto frontier with near-zero ParetoDist.

<table>
<tr>
<td width="50%">
<div align="center">
<img src="assets/Figure10-2-PanelC.png" width="100%" alt="Performance Gains and Cost Savings">
<br>
<sub>PerfGain and CostSave vs. GPT-5.</sub>
</div>
</td>
<td width="50%">
<div align="center">
<img src="assets/Figure11-1-ParetoDist.png" width="100%" alt="Pareto Frontier">
<br>
<sub>Accuracy vs. cost with Pareto frontier.</sub>
</div>
</td>
</tr>
</table>

> For more findings (embedding ablations, model pool scaling, latency analysis, etc.), please refer to our paper.

<details>
<summary><b>Original Data Tables</b></summary>

<div align="center">
<img src="assets/Table6.png" width="95%" alt="Table 6: Performance Setting">
</div>

<div align="center">
<img src="assets/Table9.png" width="95%" alt="Table 9: Performance-Cost Setting">
</div>

<div align="center">
<img src="assets/Table10.png" width="95%" alt="Table 10: Inference Costs">
</div>

</details>

---

## Installation

```bash
git clone https://github.com/ynulihao/LLMRouterBench.git
cd LLMRouterBench
pip install -r requirements.txt
```

## Quick Start

<table>
<tr>
<td>

**1. Collect Data**
```bash
python -m data_collector.cli run \
    config/data_collector_small_model_config.yaml
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



## Configurations

LLMRouterBench supports two routing paradigms with corresponding configuration files:

| Setting | Description | Collector Config | Adaptor Config |
|:---|:---|:---|:---|
| **Performance** | 20 lightweight models (~7B) | `config/data_collector_small_model_config.yaml` | `config/baseline_config.yaml` |
| **Performance-Cost** | 13 flagship models with cost | `config/data_collector_proprietary_model_config.yaml` | `config/baseline_config_performance_cost.yaml` |

## Core Components

LLMRouterBench provides a modular three-component architecture:

<div align="center">
<img src="assets/figure2-framework.png" width="95%" alt="Framework Architecture">
</div>

<table>
<tr>
<th width="33%">Collector</th>
<th width="33%">Evaluator</th>
<th width="33%">Adaptor</th>
</tr>
<tr>
<td valign="top">

Unified API interface to LLMs:
- Caching & retries
- Cost tracking
- Token counting

[Documentation](data_collector/README.md)

</td>
<td valign="top">

Dataset-specific evaluation for 21+ benchmarks:
- Dataset-Specific Modules
- Diverse Scoring logic (Regex Matching, LLM-based Judgement)
- **New**: More challenging datasets (FrontierScience, SGIBench, SFE) and multimodal support

[Documentation](evaluation/README.md)

</td>
<td valign="top">

Format conversion for 10 routing algorithms:
- Algorithm-Specific Inputs
- Consistent Train/Test Splits
- Support for ID/OOD Settings

[Documentation](baselines/README.md)

</td>
</tr>
</table>

### Customization

| Extension | Steps |
|:---|:---|
| **Add New Model** | 1. Add model config to `config/data_collector_*.yaml` 2. Run collector to generate results |
| **Add New Dataset** | 1. Create evaluator in `evaluation/` 2. Register in config files |
| **Add New Baseline** | 1. Copy algorithm to `baselines/` 2. Implement an adaptor in `baselines/adaptors/` |
---

## Datasets

LLMRouterBench includes 21 unique datasets across two settings.

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
| **Instruction Following** | ArenaHard | AHARD. | 750 | LLM as judge, 0-shot |
| **Tool Use** | τ²-Bench | TAU2. | 278 | Success Rate, 0-shot |

</details>

## Model Pools

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

## Data

LLMRouterBench stores standardized JSON records under `results/bench/` and provides download bundles for pre-collected results.

### Data Download

Download pre-collected benchmark results:

<div align="center">
<table>
<tr>
<td align="center">
<strong>Baidu Netdisk</strong><br>
<a href="https://pan.baidu.com/s/1bfa_eX3bhuo7wgNlD_dbpA?pwd=mmbf">bench-release.tar.gz</a><br>
<sub>code: mmbf</sub>
</td>
<td align="center">
<strong>Google Drive</strong><br>
<a href="https://drive.google.com/file/d/12pupoZDjqziZ2JPspH60MCC8fdXWgnX1/view?usp=drive_link">bench-release.tar.gz</a>
</td>
<td align="center">
<strong>Hugging Face</strong><br>
<a href="https://huggingface.co/datasets/NPULH/LLMRouterBench">bench-release.tar.gz</a>
</td>
</tr>
</table>
</div>

```bash
# Extract to results directory
tar xzf bench-release.tar.gz
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

## Project Structure

```
LLMRouterBench/
├── data_collector/     # Collector module
├── evaluation/         # Evaluator (21 datasets)
├── baselines/          # Adaptor & routing algorithms
├── generators/         # Model API interface
├── common/cache/       # Caching system
├── external_bench/     # Third-party integration
├── config/             # Configuration files
└── results/            # Benchmark results
```

## 🗓️ Roadmap

**Recent Updates**
- ✅ Integrated three challenging benchmarks (FrontierScience, SGIBench, SFE)
- ✅ Extended support for multimodal routing evaluation

**Long-term Goals**
- Broader model coverage
- Expanded benchmark suite
- Additional baseline methods
- Extended routing paradigms

## Related Work

### Comparison with Existing Routing Benchmarks

<div align="center">
<img src="assets/Table1_2.png" width="95%" alt="Comparison with Existing Routing Benchmarks">
<br>
</div>

Existing routing benchmarks face several limitations:

- **RouterBench**: Restricted to early-generation models and 8 relatively simple datasets.
- **EmbedLLM & RouterEval**: Focus on open-source models without inference cost information.
- **FusionFactory**: Benchmarks open-source models with estimated costs.
- **RouterArena**: Uses inconsistent model pools across routers, undermining fair comparison and lacking per-prompt, per-model data.

## Citation

If you find LLMRouterBench useful, please cite our paper:
```bibtex
@article{llmrouterbench,
  title={LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing},
  author={Li, Hao and Zhang, Yiqun and Guo, Zhaoyan and Wang, Chenxu and Tang, Shengji and Zhang, Qiaosheng and Yang, Chen and Qi, BiQing and Ye, Peng and Bai, Lei and Wang, Zhen and Hu, Shuyue},
  note={Coming soon},
  year={2025}
}
```
This work is part of our series of studies on LLM routing; if you're interested, please refer to and cite:
```bibtex
@inproceedings{zhang2025avengers,
  title        = {The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants},
  author       = {Zhang, Yiqun and Li, Hao and Wang, Chenxu and Chen, Linyao and Zhang, Qiaosheng and Ye, Peng and Feng, Shi and Wang, Daling and Wang, Zhen and Wang, Xinrun and others},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year         = {2025},
  note         = {Oral presentation},
  url          = {https://arxiv.org/abs/2505.19797}
}
@inproceedings{zhang2025beyond,
  title        = {Beyond gpt-5: Making llms cheaper and better via performance-efficiency optimized routing},
  author       = {Zhang, Yiqun and Li, Hao and Chen, Jianhao and Zhang, Hangfan and Ye, Peng and Bai, Lei and Hu, Shuyue},
  booktitle    = {Distributed AI (DAI) conference},
  year         = {2025},
  note         = {Best Paper Award},
  url          = {https://arxiv.org/abs/2508.12631}
}
@inproceedings{wang2025icl,
  title        = {ICL-Router: In-Context Learned Model Representations for LLM Routing},
  author       = {Wang, Chenxu and Li, Hao and Zhang, Yiqun and Chen, Linyao and Chen, Jianhao and Jian, Ping and Ye, Peng and Zhang, Qiaosheng and Hu, Shuyue},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year         = {2025},
  note         = {Poster},
  url          = {https://arxiv.org/abs/2510.09719}
}
@article{chen2025learning,
  title        = {Learning Compact Representations of LLM Abilities via Item Response Theory},
  author       = {Chen, Jianhao and Wang, Chenxu and Zhang, Gengrui and Ye, Peng and Bai, Lei and Hu, Wei and Qu, Yuzhong and Hu, Shuyue},
  journal      = {arXiv preprint arXiv:2510.00844},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.00844v1}
}
```

---

<div align="center">

**LLMRouterBench** — Advancing LLM Routing Research

[Report Issue](https://github.com/ynulihao/LLMRouterBench/issues) · [Request Feature](https://github.com/ynulihao/LLMRouterBench/issues)

</div>
