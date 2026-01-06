"""Utility script to sanity-check the configured embedding model.

Provide a small prompt (or a file of prompts) and verify the embedding
API returns vectors with consistent dimensionality. Example:

```
export HUOSHAN_API_KEY=...
python tools/test_embedding_model.py --text "Hello LLMRouterBench!"
```
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import yaml
from loguru import logger

from generators.factory import create_generator


def load_embedding_generator(config_path: str):
    """Create an embedding generator using the given YAML configuration."""

    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    model_config = dict(config.get("embedding_model", {}))
    cache_config = config.get("cache")

    if not model_config:
        raise ValueError(
            f"Embedding config {config_path} must define an 'embedding_model' section"
        )

    api_key = model_config.get("api_key", "")
    if api_key and api_key.isupper() and "_" in api_key:
        env_value = os.getenv(api_key)
        if not env_value:
            raise EnvironmentError(
                f"Environment variable {api_key!r} is not set but is required for the embedding model"
            )
        model_config["api_key"] = env_value

    generator = create_generator(model_config, cache_config)
    logger.info(
        "Initialised embedding generator for model %s",
        model_config.get("api_model_name", model_config.get("name")),
    )
    return generator


def iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.text:
        yield args.text
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    yield line


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the embedding generator configured in config/embedding_config.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/embedding_config.yaml",
        help="Path to embedding configuration YAML file",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single prompt to embed",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )

    args = parser.parse_args(argv)

    if not args.text and not args.file:
        parser.error("Provide --text or --file to supply prompts")

    generator = load_embedding_generator(args.config)

    success = 0
    failures = 0
    embedding_dim = None

    for prompt in iter_prompts(args):
        try:
            result = generator.generate_embedding(prompt)
            vector = result.embeddings or []
            dim = len(vector)
            if not vector:
                logger.error("Empty embedding for prompt: {}", prompt[:80])
                failures += 1
                continue

            if embedding_dim is None:
                embedding_dim = dim
            elif dim != embedding_dim:
                logger.warning(
                    "Embedding dimension changed from {} to {}; keeping first value",
                    embedding_dim,
                    dim,
                )

            logger.info(
                "Prompt: {} | dim={} | prompt_tokens={}",
                prompt[:60].replace("\n", " "),
                dim,
                getattr(result, "prompt_tokens", "?"),
            )
            success += 1
        except Exception as exc:
            logger.exception("Failed to embed prompt: %s", prompt[:80])
            failures += 1

    logger.info("Embedding test complete: {} succeeded, {} failed", success, failures)
    return 0 if failures == 0 and success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
