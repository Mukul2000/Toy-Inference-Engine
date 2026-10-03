from typing import Optional, Any
import torch
import torch.nn as nn

from inference_engine.config import ModelConfig
from inference_engine.models.rope import RotaryEmbedding
from inference_engine.models.attention import Attention
from inference_engine.models.mlp import SwiGLUMLP


class TransformerBlock(nn.Module):
    """Single Decoder Transformer Layer (Pre-Norm architecture)."""

    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(layer_idx, config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        # Only self_attn mixes rows, so only it needs `positions` and `attn_mask`;
        # the norms and MLP treat every token row independently.
        x = x + self.self_attn(
            self.input_layernorm(x), positions, attn_mask, rope, kv_cache=kv_cache
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
