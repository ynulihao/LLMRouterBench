"""Matrix Factorization inference model used by the MF router.

NOTE: This is the primary method in focus. Other routing methods
are preserved for reference but not used in the current iteration.
"""

import torch
from huggingface_hub import PyTorchModelHubMixin

MODEL_NAMES = [
    "claude-sonnet-4",
    "deepseek-v3-0324",
    "deepseek-v3.1-terminus",
    "deepseek-r1-0528",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-5-chat",
    "gpt-5",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
    "glm-4.6",
    "kimi-k2-0905",
    "intern-s1",
    "cogito-v1-preview-llama-8B",
    "DeepHermes-3-Llama-3-8B-Preview",
    "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1",
    "gemma-2-9b-it",
    "glm-4-9b-chat",
    "GLM-Z1-9B-0414",
    "granite-3.3-8b-instruct",
    "Intern-S1-mini",
    "internlm3-8b-instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1",
    "MiMo-7B-RL-0530",
    "MiniCPM4.1-8B",
    "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B",
    "Qwen3-8B",
    "Qwen2.5-Coder-7B-Instruct",
]
MODEL_IDS = {name: idx for idx, name in enumerate(MODEL_NAMES)}


class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim,
        num_classes,
        use_proj,
        embedding_model_name=None,
        embedding_generator=None,
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        if embedding_generator is None:
            raise ValueError("MFModel requires an embedding_generator instance.")

        self._embedder = embedding_generator
        self.embedding_model = (
            embedding_model_name
            or getattr(self._embedder, "model_name", None)
            or "text-embedding-3-small"
        )

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert text_dim == dim, (
                f"text_dim {text_dim} must be equal to dim {dim} if not using projection"
            )

        self.classifier = torch.nn.Linear(dim, num_classes, bias=False)

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_id, prompt):
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())

        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)

        embedding_output = self._embedder.generate_embedding(prompt)
        prompt_vector = embedding_output.embeddings
        if not prompt_vector:
            raise RuntimeError("Embedding generator returned empty embedding vector.")

        prompt_embed = torch.tensor(
            prompt_vector,
            device=self.get_device(),
            dtype=self.P.weight.dtype,
        )
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
