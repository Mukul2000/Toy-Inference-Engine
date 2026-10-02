from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference_engine.config import ModelConfig
from inference_engine.kv_cache import ContiguousKVCache
from inference_engine.models.rope import RotaryEmbedding


class Attention(nn.Module):
    """Grouped-Query Self-Attention using PyTorch's built-in SDPA."""

    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_q_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[ContiguousKVCache] = None,
    ) -> torch.Tensor:
        """
        Executes Self-Attention for the incoming tokens `x`.

        How shapes differ with vs. without `kv_cache` at Decode Step `t`:
          - WITHOUT `kv_cache` (`kv_cache=None`):
              Caller must pass the ENTIRE sequence `[0..t]`, so `num_new_tokens = t + 1`.
              `q`, `k_all`, and `v_all` all have length `t + 1`.
          - WITH `kv_cache`:
              Caller passes ONLY the 1 newest token `[t]`, so `num_new_tokens = 1`.
              1. `q`, `k_new`, `v_new` are projected for ONLY that 1 new token (length `1`).
              2. `kv_cache.update()` saves `k_new, v_new` at `positions=[t]` and returns
                 `k_all, v_all` containing the full history `[0..t]` (length `t + 1`).
              3. `q` (length `1`) attends over `k_all, v_all` (length `t + 1`).
        """
        num_new_tokens = x.shape[0]

        # 1. Project ONLY the incoming token(s) into Q, K_new, V_new
        q = self.q_proj(x).view(num_new_tokens, self.num_q_heads, self.head_dim)
        k_new = self.k_proj(x).view(num_new_tokens, self.num_kv_heads, self.head_dim)
        v_new = self.v_proj(x).view(num_new_tokens, self.num_kv_heads, self.head_dim)

        # 2. Rotate Q and K_new at their true sequence `positions`
        q, k_new = rope(q, k_new, positions)

        # 3. Assemble full K_all, V_all history (either from KV Cache or just current tokens)
        if kv_cache is not None:
            k_all, v_all = kv_cache.update(self.layer_idx, positions, k_new, v_new)
        else:
            k_all, v_all = k_new, v_new

        # 4. Scaled Dot-Product Attention: `q` [num_new_tokens] attends to `k_all, v_all` [total_seq_len]
        attn_output = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k_all.transpose(0, 1),
            v_all.transpose(0, 1),
            is_causal=(num_new_tokens > 1),
            enable_gqa=True,
        )

        # 5. Merge heads and project output
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(num_new_tokens, self.num_q_heads * self.head_dim)
        )
        return self.o_proj(attn_output)
