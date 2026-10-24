from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from inference_engine.config import ModelConfig
from inference_engine.executor.batch_metadata import BatchMetadata
from inference_engine.executor.kv_cache import ContiguousKVCache
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
        meta: BatchMetadata,
        rope: RotaryEmbedding,
        kv_cache: Optional[ContiguousKVCache] = None,
    ) -> torch.Tensor:
        """
        Executes Self-Attention for the N packed tokens `x` of all rows at once
        (design_doc.md §7.6-7.7).

        Shapes (N = packed tokens; per group: b = rows, t = its longest row, s = key columns):
          - x:              [N, hidden_size]
          - meta.positions: [N]  RoPE position (= column in its slot) of each token
          - meta.groups:    one rectangle per group: chunk rows, then decode rows

        How shapes differ with vs. without `kv_cache` at Decode Step `t`:
          - WITHOUT `kv_cache` (`kv_cache=None`):
              Caller must pass the ENTIRE sequence `[0..t]` of every row; keys = those tokens.
          - WITH `kv_cache`:
              Caller passes ONLY the tokens not yet in the cache (a decode row: its newest one).
              1. `q`, `k_new`, `v_new` are projected for ONLY those tokens.
              2. `kv_cache.write()` saves `k_new, v_new` at each token's column of its slot.
              3. Per group, `kv_cache.read()` returns every row's history from its OWN slot,
                 and each row's queries attend over it.
        """
        # `rearrange` patterns name every axis: n = packed tokens, b = rows, t = new tokens
        # per row, s = all keys attended over (cached + new), heads, d = head_dim.

        # 1. Project ONLY the incoming token(s), then split the last dim into heads.
        #    Linear layers work token by token: they don't care which request a token is from.
        q = rearrange(self.q_proj(x), "n (heads d) -> n heads d", d=self.head_dim)
        k_new = rearrange(self.k_proj(x), "n (heads d) -> n heads d", d=self.head_dim)
        v_new = rearrange(self.v_proj(x), "n (heads d) -> n heads d", d=self.head_dim)

        # 2. Rotate Q and K_new at their true per-request `positions`
        q, k_new = rope(q, k_new, meta.positions)

        # 3. Save the new tokens' K/V into their slots, before any group reads them
        if kv_cache is not None:
            kv_cache.write(self.layer_idx, k_new, v_new, meta.slots, meta.positions)

        # 4. Scaled Dot-Product Attention, one rectangle per group (chunk rows, then decode
        #    rows), separately for each row and head. SDPA wants heads before tokens. The
        #    explicit mask replaces `is_causal`, which assumes every row starts at column 0.
        outputs = []
        for group in meta.groups:
            if kv_cache is not None:  # each row's full history, from its own slot
                k_all, v_all = kv_cache.read(self.layer_idx, group.slot_ids, group.seq_len)
            else:  # fresh rows from column 0: the history is just the new tokens
                k_all = group.to_rect(k_new[group.tokens])
                v_all = group.to_rect(v_new[group.tokens])
            attn_output = F.scaled_dot_product_attention(
                rearrange(group.to_rect(q[group.tokens]), "b t heads d -> b heads t d"),
                rearrange(k_all, "b s heads d -> b heads s d"),
                rearrange(v_all, "b s heads d -> b heads s d"),
                attn_mask=group.attn_mask,
                enable_gqa=True,
            )
            outputs.append(group.from_rect(rearrange(attn_output, "b heads t d -> b t heads d")))

        # 5. Pack the groups back together (they are contiguous slices, in order), glue the
        #    heads back together, then mix them with W_o
        return self.o_proj(rearrange(torch.cat(outputs), "n heads d -> n (heads d)"))
