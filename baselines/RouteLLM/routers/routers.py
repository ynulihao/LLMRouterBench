import abc
import os
from pathlib import Path

import torch
import yaml

from generators.factory import create_generator
from generators.generator import EmbeddingGenerator

from .matrix_factorization.model import MFModel, MODEL_IDS

def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]

@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        # this is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model: str,
        weak_model: str,
        hidden_size: int = 128,
        num_models: int = 64,
        text_dim: int = 1536,
        num_classes: int = 1,
        use_proj: bool = True,
        embedding_config_path: str | None = "config/embedding_config.yaml",
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embedder, embedding_model_name = self._load_embedding_generator(embedding_config_path)

        checkpoint_file = self._resolve_checkpoint_path(checkpoint_path)
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("Q.")
        }

        model = MFModel(
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
            embedding_model_name=embedding_model_name,
            embedding_generator=embedder,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {unexpected}"
            )
        if missing:
            missing_filtered = [key for key in missing if not key.startswith("Q.")]
            if missing_filtered:
                raise RuntimeError(
                    f"Missing required keys when loading checkpoint: {missing_filtered}"
                )

        self.model = model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]
        self.embedding_model_name = embedding_model_name
        self.embedding_generator = embedder
    
    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate

    @staticmethod
    def _load_embedding_generator(config_path: str | None):
        if not config_path:
            raise ValueError("MatrixFactorizationRouter requires an embedding_config_path.")

        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding configuration file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}

        model_cfg = dict(config.get("embedding_model") or {})
        cache_cfg = config.get("cache")

        if not model_cfg:
            raise ValueError("embedding_config.yaml must define 'embedding_model'.")

        model_cfg.setdefault("generator_type", "embedding")

        api_key = model_cfg.get("api_key") or ""
        if api_key and api_key.isupper() and "_" in api_key:
            model_cfg["api_key"] = os.getenv(api_key, api_key)

        embedder = create_generator(model_cfg, cache_cfg)
        if not isinstance(embedder, EmbeddingGenerator):
            raise TypeError(
                "Embedding configuration must produce an EmbeddingGenerator. "
                f"Received: {type(embedder).__name__}"
            )

        embedding_model = (
            model_cfg.get("api_model_name")
            or model_cfg.get("name")
            or embedder.model_name
        )

        return embedder, embedding_model

    @staticmethod
    def _resolve_checkpoint_path(path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate

        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {candidate}")

        for name in ("pytorch_model.bin", "mf_model.pt", "model.pt"):
            guess = candidate / name
            if guess.exists():
                return guess

        raise FileNotFoundError(
            f"No checkpoint file found under {candidate}. "
            "Expected one of: pytorch_model.bin, mf_model.pt, model.pt"
        )

ROUTER_CLS = {
    "mf": MatrixFactorizationRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
