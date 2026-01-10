# SGI-Bench 评测指南

SGI-Bench (Scientific General Intelligence Benchmark) 是一个评估大语言模型科学研究能力的基准测试。本项目实现了 SGI-Bench 的 4 个子任务，完全对齐官方评测逻辑。

## 子任务概览

| 子任务 | 样本数 | 评测方式 | 核心指标 |
|--------|--------|----------|----------|
| Deep Research | 318 | LLM Judge | Exact Match, Step-Level Accuracy |
| Idea Generation | 315 | 3模型投票 + 嵌入相似度 | 4维度综合分 |
| Dry Experiment | 271 | 代码执行 | PassAll@5/3/1, AET, SER |
| Wet Experiment | 68 | 序列比对 | Action Similarity, Parameter Accuracy |

---

## 1. Deep Research (深度研究)

**任务描述:** 多跳科学问答，需要信息检索和计算推理。

### 评测流程

```
模型输出 → 提取<answer>标签 → AnswerParser标准化 → Exact Match比对
                                                  ↓
                                            LLM Judge评判步骤正确性
```

### 核心指标

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | `answer == extracted_answer OR answer == parsed_answer` |
| **Step-Level Accuracy (SLA)** | LLM 逐步评判正确率 |

### 关键实现

- **AnswerParser** (`utils.py:AnswerParser`): 使用 `gpt-4.1-mini` 将答案标准化
  - 数字答案使用示例: `"0.25"`
  - 文本答案使用示例: `"T cell and B cell"`
  - **注意:** 不传入真实答案，避免泄露

### 代码入口

```python
# evaluation/SGIBench/deep_research.py
class DeepResearchEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 2. Idea Generation (想法生成)

**任务描述:** 给定研究背景，生成科研想法/假设。

### 评测流程

```
模型输出(JSON) → 解析6个字段 → 客观分计算(4维度) → 主观投票(3模型×2票)
                                      ↓                      ↓
                              embedding相似度           win/lose/tie判定
                              graph相似度                    ↓
                                      ↓              win=100, else=0
                              objective_score              ↓
                                      └──────→ (obj + subj) / 2 ──→ 最终分
```

### 核心指标

| 维度 | 客观分计算 | 主观分判定 |
|------|-----------|-----------|
| **Effectiveness** | 关键词嵌入相似度 × 100 | votes > 3 → win |
| **Novelty** | 0.5×novelty_sim + 0.5×cutting_edge | votes > 4 → win |
| **Detailedness** | 0.2×完整度 + 0.4×(10-重复) + 0.4×(10-长度) | votes > 3 → win |
| **Feasibility** | graph_similarity × 100 | votes > 3 → win |

**最终分 = (Effectiveness + Novelty + Detailedness + Feasibility) / 4**

### 投票机制

- **3个Judge模型** × **2次投票**(正常+交换位置) = **6票**
- Win Gate: novelty > 4票, 其他 > 3票
- 主观分: win=100, tie/lose=0

### 代码入口

```python
# evaluation/SGIBench/idea_generation.py
class IdeaGenerationEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 3. Dry Experiment (代码实验)

**任务描述:** 补全科学计算的 Python 函数代码。

### 评测流程

```
模型输出 → 提取代码 → 替换到主程序 → 执行5个单元测试 → 输出比对
                                            ↓
                                    精确匹配 OR LLM判断
```

### 核心指标

| 指标 | 说明 |
|------|------|
| **PassAll@5** | 5个测试全部通过 |
| **PassAll@3** | 至少3个测试通过 |
| **PassAll@1** | 至少1个测试通过 |
| **AET** | 平均执行时间 |
| **SER** | 执行成功率 (成功次数/5) |

### 特殊要求

1. **Conda环境**: 必须存在 `dryexp` 环境 (可通过 `SGI_CONDA_ENV` 覆盖)
2. **数据Fixtures**: 样本 0200, 0206, 0236 需要额外数据文件

### 环境配置 (Dry Experiment 专用)

```bash
# Step 1: 创建 Conda 环境
conda create -n dryexp python=3.10 -y
conda activate dryexp

# Step 2: 安装依赖
pip install -r evaluation/SGIBench/dryexp_requirements.txt

# Step 3: 配置环境变量
export SGI_CONDA_ENV="dryexp"
export SGI_DRY_EXPERIMENT_DATA_DIR="/path/to/data/fixtures"  # 样本0200/0206/0236需要
```

### 代码入口

```python
# evaluation/SGIBench/dry_experiment.py
class DryExperimentEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 4. Wet Experiment (湿实验)

**任务描述:** 生成实验室实验协议的伪代码。

### 评测流程

```
模型输出 → 解析步骤 → 动作序列比对 → Kendall tau距离
                  ↓
            参数映射比对 → 参数准确率
```

### 核心指标

| 指标 | 说明 |
|------|------|
| **Action Sequence Similarity** | Kendall tau 序列相似度 |
| **Parameter Accuracy** | 参数匹配准确率 |
| **Final Score** | (序列相似度 + 参数准确率) / 2 |

### 代码入口

```python
# evaluation/SGIBench/wet_experiment.py
class WetExperimentEvaluator(SGIBenchBaseEvaluator):
    def evaluate(self, data, output_text, **kwargs) -> Dict
```

---

## 代码结构

```
evaluation/SGIBench/
├── __init__.py              # 导出4个评测器
├── base.py                  # SGIBenchBaseEvaluator 基类
├── prompts.py               # Prompt模板
├── utils.py                 # 工具函数 (AnswerParser, graph_similarity等)
├── deep_research.py         # DeepResearchEvaluator
├── idea_generation.py       # IdeaGenerationEvaluator
├── dry_experiment.py        # DryExperimentEvaluator
├── wet_experiment.py        # WetExperimentEvaluator
└── dryexp_requirements.txt  # Dry Experiment Conda环境依赖 (68个包)

data/sgibench/
├── deep_research/test.jsonl    # 318 samples
├── idea_generation/test.jsonl  # 315 samples
├── dry_experiment/test.jsonl   # 271 samples
└── wet_experiment/test.jsonl   # 68 samples
```

---

## 运行指南

### 快速开始

```bash
# 1. 编辑配置文件，启用需要的子任务
vim config/data_collector_sgibench.yaml

# 2. 运行评测
./run_sgibench.sh

# 或指定自定义配置文件
./run_sgibench.sh config/my_custom.yaml
```

### 环境变量配置

#### 必需配置

```bash
# OpenRouter API (模型推理)
export OPENROUTER_API_KEY="your-api-key"

# Grader API (共享凭据，用于所有需要 LLM Judge 的子任务)
export GRADER_BASE_URL="https://openrouter.ai/api/v1"
export GRADER_API_KEY="$OPENROUTER_API_KEY"
```

#### Deep Research 配置

```bash
# Grader模型 (用于LLM Judge和AnswerParser)
# 官方使用 o4-mini, temperature=0
export SGI_DR_GRADER_MODEL="o4-mini"
```

#### Dry Experiment 配置

```bash
# Grader模型 (用于LLM Judge判断输出是否正确)
# 官方使用 o4-mini, temperature=0
export SGI_DE_GRADER_MODEL="o4-mini"

# Conda环境 (必须存在)
export SGI_CONDA_ENV="dryexp"                           # 官方默认

# 执行超时
export SGI_EXECUTION_TIMEOUT="300"                      # 5分钟

# 数据Fixtures目录 (样本0200/0206/0236需要)
export SGI_DRY_EXPERIMENT_DATA_DIR="/path/to/fixtures"
```

#### Idea Generation 配置

```bash
# 必须配置3个Judge模型 (官方要求，temperature=0.1)
export SGI_JUDGE_MODEL_1="gpt-5.1-2025-11-13"           # Judge 1
export SGI_JUDGE_MODEL_2="gemini-3-pro-preview"         # Judge 2
export SGI_JUDGE_MODEL_3="anthropic/claude-sonnet-4.5"  # Judge 3

export SGI_JUDGE_BASE_URL_1="https://openrouter.ai/api/v1"
export SGI_JUDGE_BASE_URL_2="https://openrouter.ai/api/v1"
export SGI_JUDGE_BASE_URL_3="https://openrouter.ai/api/v1"

export SGI_JUDGE_API_KEY_1="$OPENROUTER_API_KEY"
export SGI_JUDGE_API_KEY_2="$OPENROUTER_API_KEY"
export SGI_JUDGE_API_KEY_3="$OPENROUTER_API_KEY"

# 嵌入模型 (客观分计算)
export SGI_EMBEDDING_MODEL="all-MiniLM-L6-v2"           # 官方默认
```

#### Wet Experiment 配置

```bash
# Wet Experiment 使用规则评分，不需要 LLM Judge
#
# 注意：官方实现在 step_1_get_answer.py 中调用 AnswerParser，但在 step_2_score.py
# 评分时并未使用 parser 结果（line 241 被注释掉），而是直接用原始答案。
# 因此本实现也不调用 AnswerParser，与官方评分逻辑一致。
```

### 使用配置文件运行

```bash
# 编辑配置文件
vim config/data_collector_sgibench.yaml

# 运行
python -m data_collector.cli run config/data_collector_sgibench.yaml
```

### 配置文件示例

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
  # - dataset_id: sgibench-ideageneration  # 需要3个judge模型
  # - dataset_id: sgibench-dryexperiment   # 需要dryexp conda环境

run:
  output_dir: ./results
  concurrency: 8
  demo_mode: true
  demo_limit: 10
```

---

## 官方对齐说明

本实现完全对齐官方 SGI-Bench 评测逻辑:

### Grader 配置对齐

| 子任务 | 官方 Grader 模型 | temperature | 环境变量 |
|--------|-----------------|-------------|----------|
| Deep Research | `o4-mini` | 0 | `SGI_DR_GRADER_MODEL` |
| Dry Experiment | `o4-mini` | 0 | `SGI_DE_GRADER_MODEL` |
| Idea Generation | 3个 Judge 模型 | 0.1 | `SGI_JUDGE_MODEL_1/2/3` |
| Wet Experiment | 无需 LLM Judge | - | - |

### 评测逻辑对齐

| 对齐项 | 官方实现 | 本地实现 |
|--------|----------|----------|
| DR/DE Grader | `o4-mini`, temp=0 | ✅ 相同 |
| IG Judge模型 | `["gpt-5.1-2025-11-13", "gemini-3-pro-preview", "anthropic/claude-sonnet-4.5"]` | ✅ 相同 |
| IG temperature | 0.1 | ✅ 相同 |
| AnswerParser示例 | `"0.25"` / `"T cell and B cell"` | ✅ 相同 |
| 投票Win Gate | novelty>4, others>3 | ✅ 相同 |
| 主观分 | win=100, else=0 | ✅ 相同 |
| graph_similarity | 顺序边fallback | ✅ 相同 |
| Dry Experiment环境 | `dryexp` conda | ✅ 强制校验 |
| DE data_en.py执行 | step_1_build.py 预先运行 | ✅ 每次测试前运行 |
| DE data_en.py超时 | 10分钟 | ✅ 相同 |

---

## 依赖要求

```bash
# 基础依赖 (Idea Generation)
pip install sentence-transformers  # 嵌入计算
pip install networkx               # 图相似度
pip install json-repair            # JSON解析修复 (可选)

# Dry Experiment 环境
conda create -n dryexp python=3.10 -y
conda activate dryexp
pip install -r evaluation/SGIBench/dryexp_requirements.txt
```

---

## 参考

- 官方代码: `SGI-Bench/evaluation/`
- 论文: SGI-Bench: A Benchmark for Scientific General Intelligence
