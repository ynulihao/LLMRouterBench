# 🔍 Evaluator

The **Evaluator** module provides unified evaluation for 27+ benchmarks with consistent interfaces, answer extraction, and scoring mechanisms.

---

## 📊 Supported Datasets

| Category | Datasets |
|:---|:---|
| **Math** | AIME, MATH500, MATHBench, LiveMathBench, FinQA |
| **Code** | HumanEval, MBPP, LiveCodeBench, StudentEval |
| **Reasoning** | BBH, ARC-C, Winogrande, GPQA, MMLU-Pro, K&K, KORBench, BrainTeaser |
| **Knowledge QA** | SimpleQA, MedQA, TruthfulQA, SFE, HLE |
| **Emotion/NLP** | EmoryNLP, MELD, DailyDialog |
| **Benchmarking** | ArenaHard |
| **Vision** | ARC-AGI |

---

## 🚀 Quick Start

```python
from evaluation import EvaluatorFactory

# Create factory
factory = EvaluatorFactory(max_workers=8)

# Get evaluator for a specific benchmark
evaluator = factory.get_evaluator("mmlu_pro")

# Load data
data = evaluator.load_data(split="test")

# Evaluate a single sample
result = evaluator.evaluate(
    data=data[0],
    output_text="The answer is A."
)
# Returns: {"prediction": "A", "ground_truth": "A", "is_correct": True}
```

---

## 🔌 Adding New Evaluators

### Step 1: Create Directory Structure

```
evaluation/MyBenchmark/
├── __init__.py
├── mybenchmark.py
└── prompts.py (optional)
```

### Step 2: Implement Evaluator Class

```python
# evaluation/MyBenchmark/mybenchmark.py
import os
from typing import Dict, List, Optional, Any
from datasets import Dataset, disable_progress_bars
from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/MyBenchmark"

PROMPT = """Answer the following question.

Question: {question}

Answer:"""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class MyBenchmarkEvaluator(BaseEvaluator):
    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        super().__init__(grader_cache_config)
        self.task = "MyBenchmark"

    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
        return data

    def get_valid_splits(self) -> List[str]:
        return ["test"]

    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(question=item["question"])
        return {"prompt": prompt}

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        prediction = self.extract_normal_answer(output_text, ANSWER_PATTERN)
        ground_truth = data['answer']
        is_correct = prediction == ground_truth

        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        }
```

### Step 3: Create `__init__.py`

```python
# evaluation/MyBenchmark/__init__.py
from evaluation.MyBenchmark.mybenchmark import MyBenchmarkEvaluator

__all__ = ["MyBenchmarkEvaluator"]
```

### Step 4: Register in Factory

```python
# evaluation/factory.py

# 1. Add to Benchmark enum
class Benchmark(Enum):
    # ... existing benchmarks ...
    MYBENCHMARK = 'mybenchmark'

# 2. Add import
from evaluation.MyBenchmark import MyBenchmarkEvaluator

# 3. Add to get_evaluator method
def get_evaluator(self, task: str | Benchmark):
    # ... existing conditions ...
    elif task == Benchmark.MYBENCHMARK:
        return MyBenchmarkEvaluator(grader_cache_config=self.grader_cache_config)
```

---

## 📋 Evaluator Patterns

### Pattern 1: Math/Logic (Regex + Symbolic Grading)

For math problems with LaTeX answers:

```python
from evaluation.deepscaler_rm import extract_answer, grade_answer_mathd, grade_answer_sympy

def evaluate(self, data, output_text, **kwargs):
    prediction = extract_answer(output_text)
    ground_truth = extract_answer(data['Answer'])

    is_correct = grade_answer_mathd(prediction, ground_truth) or \
                 grade_answer_sympy(prediction, ground_truth)

    return {"prediction": prediction, "ground_truth": ground_truth, "is_correct": is_correct}
```

### Pattern 2: Code Generation (Execution-Based)

For code problems with test cases:

```python
from evaluation.HumanEval.execution import check_correctness

def evaluate(self, data, output_text, **kwargs):
    code = self.extract_code(output_text)
    result = check_correctness(
        task_id=data['task_id'],
        completion_id=0,
        solution=code,
        time_out=10
    )

    return {"prediction": code, "ground_truth": data['canonical_solution'], "is_correct": result['passed']}
```

### Pattern 3: Multiple Choice (Pattern Matching)

For MCQ with option extraction:

```python
def evaluate(self, data, output_text, **kwargs):
    prediction = self.extract_normal_answer(
        text=output_text,
        answer_pattern=r"(?i)Answer\s*:\s*\$?([A-J])[.\s\n]?"
    )
    is_correct = data['answer'] == prediction

    return {"prediction": prediction, "ground_truth": data['answer'], "is_correct": is_correct}
```

### Pattern 4: LLM-Based Grading (Semantic Evaluation)

For open-ended questions requiring semantic comparison:

```python
GRADER_TEMPLATE = """Question: {question}
Target Answer: {target}
Model Answer: {predicted_answer}

Grade: A (Correct), B (Partially Correct), C (Incorrect)"""

def evaluate(self, data, output_text, **kwargs):
    prompt = GRADER_TEMPLATE.format(
        question=data['problem'],
        target=data['answer'],
        predicted_answer=output_text
    )
    result = self.grader.generate(prompt)
    grade = re.search(r"(A|B|C)", result.output).group(0)

    return {"prediction": grade, "ground_truth": data['answer'], "is_correct": grade == "A"}
```

---

## 🔧 BaseEvaluator Utilities

| Method | Description |
|:---|:---|
| `load_jsonl(path)` | Load JSONL/JSON data files |
| `extract_boxed_content(text)` | Extract `\boxed{}` LaTeX content |
| `extract_normal_answer(text, pattern)` | Regex-based answer extraction |
| `self.grader` | LLM grader for semantic evaluation |

---

## 🗂️ File Structure

```
evaluation/
├── __init__.py              # Exports EvaluatorFactory
├── base_evaluator.py        # BaseEvaluator abstract class
├── factory.py               # EvaluatorFactory + Benchmark enum
├── deepscaler_rm.py         # Math grading utilities
├── AIME/
│   ├── __init__.py
│   └── aime.py
├── HumanEval/
│   ├── __init__.py
│   ├── humaneval.py
│   └── execution.py
├── MMLUPro/
│   ├── __init__.py
│   └── mmlupro.py
└── ...
```
