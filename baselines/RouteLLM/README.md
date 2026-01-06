# RouteLLM: Learning to Route LLMs with Preference Data

This baseline integrates [RouteLLM](https://github.com/lm-sys/RouteLLM)'s matrix factorization router for LLM routing in LLMRouterBench.

**Paper**: [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)

## Usage

### Step 1: Generate Training Data

```bash
python -m baselines.adaptors.routellm_adaptor \
    --config config/baseline_config.yaml \
    --strong-model gpt-5 \
    --weak-model gemini-2.5-flash \
    --output-dir baselines/RouteLLM/data
```

### Step 2: Train Router

```bash
python -m baselines.RouteLLM.routers.matrix_factorization.train_matrix_factorization \
    --config baselines/RouteLLM/mf_train_config.json
```

### Step 3: Evaluate Router

```bash
python -m baselines.RouteLLM.evaluate_mf \
    --config baselines/RouteLLM/router_eval_config.json \
    --data-dir baselines/RouteLLM/data/seed42_split0.8_gpt-5__vs__gemini-2.5-flash \
    --strong-model gpt-5 \
    --weak-model gemini-2.5-flash \
    --threshold 0.5
```

## Citation

```bibtex
@article{ong2024routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E and Kadous, M Waleed and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.18665},
  year={2024}
}
```
