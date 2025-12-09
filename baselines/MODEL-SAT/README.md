This directory contains the reproduced code for [Capability Instruction Tuning: A New Paradigm for Dynamic LLM Routing](https://arxiv.org/abs/2502.17282).

### Installation

```
conda create -n model_sat python=3.10

pip install -r requirements.txt
```

### Construct Dataset

```
sh ./scripts/construct_dataset.sh
```

### Generate Model Descriptions

```
sh ./scripts/generate_model_description.sh
```

There are five datasets, each created using a different random seed. The default is seed42. If you need to use a different random seed, please modify **construct_dataset.sh** and **generate_model_description.sh**.

### Train

```
sh train.sh
```

The evaluation is conducted every 2,000 training steps.