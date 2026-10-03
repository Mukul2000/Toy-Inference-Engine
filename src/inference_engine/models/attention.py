from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        attn_mask: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[ContiguousKVCache] = None,
    ) -> torch.Tensor:
        """
        Executes Self-Attention for the incoming tokens `x` of all B requests at once.

        Shapes (B = batch size, T = new tokens per row, S = total columns attended over):
          - x:         [B, T, hidden_size]
          - positions: [B, T]        RoPE position of each token within its own request
          - attn_mask: [B, 1, T, S]  True = may attend (causal, never a pad, never another request)

        How shapes differ with vs. without `kv_cache` at Decode Step `t`:
          - WITHOUT `kv_cache` (`kv_cache=None`):
              Caller must pass the ENTIRE sequence `[0..t]` of every row, so `T = S = t + 1`.
          - WITH `kv_cache`:
              Caller passes ONLY each row's newest token, so `T = 1`.
              1. `q`, `k_new`, `v_new` are projected for ONLY that 1 new token per row.
              2. `kv_cache.update()` saves `k_new, v_new` in the next free column and returns
                 `k_all, v_all` containing every row's full history (`S = t + 1` columns).
              3. `q` (1 per row) attends over `k_all, v_all` of its OWN row.
        """
        # `rearrange` patterns name every axis: b = batch, t = new tokens, s = all keys
        # attended over (cached + new), heads, d = head_dim.

        # 1. Project ONLY the incoming token(s), then split the last dim into heads
        q = rearrange(self.q_proj(x), "b t (heads d) -> b t heads d", d=self.head_dim)
        k_new = rearrange(self.k_proj(x), "b t (heads d) -> b t heads d", d=self.head_dim)
        v_new = rearrange(self.v_proj(x), "b t (heads d) -> b t heads d", d=self.head_dim)

        # 2. Rotate Q and K_new at their true per-request `positions`
        q, k_new = rope(q, k_new, positions)

        # 3. Assemble full K_all, V_all history (either from KV Cache or just current tokens)
        if kv_cache is not None:
            k_all, v_all = kv_cache.update(self.layer_idx, k_new, v_new)
        else:
            k_all, v_all = k_new, v_new

        # 4. Scaled Dot-Product Attention, separately for each request and head. SDPA wants
        #    heads before tokens. The explicit mask replaces `is_causal`, which knows nothing
        #    about pads.
        attn_output = F.scaled_dot_product_attention(
            rearrange(q, "b t heads d -> b heads t d"),
            rearrange(k_all, "b s heads d -> b heads s d"),
            rearrange(v_all, "b s heads d -> b heads s d"),
            attn_mask=attn_mask,
            enable_gqa=True,
        )

        # 5. Glue the heads back together, then mix them with W_o
        return self.o_proj(rearrange(attn_output, "b heads t d -> b t (heads d)"))
