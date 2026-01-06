# 🔌 External Benchmark Integration

A lightweight API wrapper for integrating complex third-party benchmarks (SWE-bench, τ²-bench, etc.) that are too difficult to port into LLMRouterBench's standard evaluation system.

---

## ✨ Why external_bench?

Some benchmarks are **too complex to integrate** into the standard pipeline:
- **SWE-bench**: Multi-step agent workflows, code execution environments
- **τ²-bench**: Complex multi-turn interactions, specialized environments

**external_bench provides minimal-touch integration**:
- Use LLMRouterBench's unified API management (`DirectGenerator`)
- Benefit from MySQL caching to reduce API costs
- Keep your entire evaluation system unchanged

---

## 🚀 Quick Start

### Step 1: Add Imports

```python
from external_bench import setup, DirectGenerator, RecordResult, finish_benchmark, start_timer
setup()
```

### Step 2: Initialize Generator

```python
generator = DirectGenerator(
    model_name="anthropic/claude-3.5-sonnet",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.anthropic.com/v1",
    cache_config=cache_config  # Optional
)
```

### Step 3: Replace API Calls

```python
# Before
def query_model(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# After
def query_model(prompt):
    result = generator.generate(prompt)
    return result.output
```

### Step 4: Collect Results

```python
start_timer()
record_results = []

for i, task in enumerate(tasks):
    response = query_model(build_prompt(task))
    is_correct = evaluate_response(response, task)

    record_results.append(RecordResult(
        index=i,
        origin_query=task['question'],
        prompt=prompt,
        prompt_tokens=generator.last_prompt_tokens,
        completion_tokens=generator.last_completion_tokens,
        cost=generator.last_cost,
        score=1.0 if is_correct else 0.0,
        prediction=response,
        ground_truth=task['answer'],
        raw_output=response
    ))
```

### Step 5: Save Results

```python
accuracy = finish_benchmark(record_results, "gpt-4", "your_benchmark", "test")
```

---

## 📖 API Reference

### DirectGenerator

```python
DirectGenerator(
    model_name: str,           # API model identifier
    api_key: str,              # API key or env var name
    base_url: str,             # API endpoint URL
    temperature: float = 0.7,
    max_tokens: int = 2048,
    cache_config: dict = None  # MySQL cache config
)

result = generator.generate(prompt: str)
# Returns: GeneratorOutput (output, prompt_tokens, completion_tokens, cost)
```

### RecordResult

```python
RecordResult(
    index: int,
    origin_query: str,
    prompt: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    score: float,              # 1.0 = correct, 0.0 = incorrect
    prediction: str,
    ground_truth: str,
    raw_output: str
)
```

### finish_benchmark

```python
accuracy = finish_benchmark(
    record_results: List[RecordResult],
    model_name: str,
    dataset_name: str = "external_bench",
    split: str = "test",
    base_dir: str = "results/bench/external_bench"
)
```

---

## 📄 Result Storage

Results saved to: `results/bench/external_bench/<dataset>/<split>/<model>/<timestamp>.json`

Format matches LLMRouterBench's standard for compatibility with baselines and analysis tools.

---

## ✅ Integration Checklist

- [ ] Import `DirectGenerator`, `RecordResult`, `finish_benchmark`, `start_timer`
- [ ] Replace API calls with `generator.generate()`
- [ ] Add `RecordResult` collection after each evaluation
- [ ] Call `start_timer()` at benchmark start
- [ ] Call `finish_benchmark()` at benchmark end
- [ ] (Optional) Enable MySQL caching

**Do NOT**:
- Refactor benchmark structure
- Change evaluation metrics or logic
- Modify workflow orchestration
