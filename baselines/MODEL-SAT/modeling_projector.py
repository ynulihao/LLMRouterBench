import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel, PretrainedConfig


class ProjectorConfig(PretrainedConfig):
    def __init__(self, in_features=4096, out_features=None, expansion_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.expansion_ratio = expansion_ratio

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class MLPProjector(PreTrainedModel):
    """
    A simple 2-layer FFN projector without gating (≈ standard Transformer FFN).
    """
    def __init__(self, config: ProjectorConfig, dtype = torch.float32):
        super().__init__(config)
        in_features, out_features, ratio = (
            config.in_features,
            config.out_features or config.in_features,
            config.expansion_ratio,
        )
        bias = False

        # ===== modules =====
        self.pre_norm  = LlamaRMSNorm(in_features)          # Pre-LN
        self.input_fc  = nn.Linear(in_features, out_features, bias=bias)

        hidden = out_features * ratio                       # expansion dim
        self.fc1 = nn.Linear(out_features, hidden, bias=bias)
        self.act = nn.SiLU()                                # or GELU
        self.fc2 = nn.Linear(hidden, out_features, bias=bias)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.input_fc(x)

        # Feed-Forward block (no gating)
        y = self.act(self.fc1(x))
        y = self.fc2(y)

        x = x + y     
        return x


class LinearProjector(PreTrainedModel):
    def __init__(self, config: ProjectorConfig, dtype=torch.float32, **kwargs):
        super().__init__(config)
        in_features, out_features = config.in_features, config.out_features
        self.config = config
        if out_features is None:
            out_features = in_features
        self.linear = nn.Linear(in_features, out_features, dtype=dtype, bias=False)

    def forward(self, x):
        return self.linear(x)


class ModelWrapper(PreTrainedModel):
    def __init__(self, model: PreTrainedModel, config: PretrainedConfig = None, max_length=512, tokenizer=None):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, text_lst, **kwargs):
        input_ids = self.tokenizer(text_lst, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)['input_ids'].to(self.model.device)
        return self._encode(input_ids, **kwargs)
    
    def _encode(self, input_ids, attention_mask=None, **kwargs):
        # return the last hidden states
        return self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].mean(dim=1, keepdim=True)

