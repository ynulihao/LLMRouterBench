"""Routing controller.

NOTE: This iteration prioritizes the Matrix Factorization (MF) router.
Other router implementations are kept for reference but are not the focus.
"""

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from .routers.routers import ROUTER_CLS

# Default configuration focused on MF only.
GPT_4_AUGMENTED_CONFIG = {
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}

class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str
    
class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        self.model_pair = ModelPair(strong_model, weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        
        if config is None:
            config = GPT_4_AUGMENTED_CONFIG
        
        router_pbar = None
        if self.progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()
        
        # NOTE: We focus on the matrix factorization ("mf") router in this
        # iteration. Other routers are retained but not actively used.
        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            router_kwargs = dict(config.get(router, {}))
            # Only MF requires explicit strong/weak model IDs on init here.
            if router == "mf":
                router_kwargs.update({
                    "strong_model": strong_model,
                    "weak_model": weak_model,
                })
            self.routers[router] = ROUTER_CLS[router](**router_kwargs)
        
        # some python magic to match the openai python sdk
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )
    
    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(f"Router {router} not found.")
        if not 0 <= threshold <= 1:
            raise RoutingError(f"Invalid threshold {threshold}. Must be in [0, 1].")
    
    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)

        self.model_counts[router][routed_model] += 1

        return routed_model
    
    # Mainly used for evaluationss
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)

        return self.routers[router].route(prompt, threshold, self.model_pair)
    
    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return completion(api_base=self.api_base, api_key=self.api_key, **kwargs)

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)
