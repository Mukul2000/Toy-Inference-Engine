import torch
from einops import rearrange

from inference_engine.config import ModelConfig


class ContiguousKVCache:
    """
    The KV slot pool for continuous batching (design_doc.md §6.3).

    One pre-allocated buffer each for K and V:
        [num_layers, num_slots, max_seq_len, num_kv_heads, head_dim]

    A request owns one slot from admission until it finishes. Inside its slot, its tokens
    sit at columns 0, 1, 2, ... in order, so column = position.

    Storage only: the pool keeps no lengths and no cursor. Every call says which slots to
    write and from which column (`slot_ids`, `starts`); Layer 1 decides both. A freed slot
    is never cleared: its next owner overwrites it from column 0, and the `k <= q` mask
    (see `BatchMetadata`) never lets a request read past its own length.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_slots: int,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len

        # It's space set aside before generation starts, not what actually gets used.
        cache_shape = (
            config.num_hidden_layers, # cache for which layer index
            num_slots, # which slot: each running request owns one, and freed slots are reused
            max_seq_len, # capacity of ONE slot: a request's prompt + max_new_tokens must fit
            config.num_key_value_heads, # number of key/value heads
            config.head_dim, # dimension of each head
            # note that the last 2 make up the output width of k_proj / v_proj:
            # num_kv_heads * head_dim = 3 * 64 = 192 for SmolLM2. Because of GQA that is
            # smaller than the hidden dimension (9 query heads * 64 = 576).
        )
        # the last 3 parameters match one request's k_new / v_new ([T, num_kv_heads, head_dim],
        # the k_proj / v_proj outputs split into heads), with T widened to max_seq_len.
        self.k_cache = torch.zeros(cache_shape, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype)

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        slot_ids: torch.Tensor,
        starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. Scatter: writes row b of `k_new, v_new` ([B, T, num_kv_heads, head_dim]) into
           columns starts[b] .. starts[b] + T - 1 of slot slot_ids[b].
        2. Gather: returns every row's history `k_all, v_all` ([B, S, num_kv_heads, head_dim]),
           columns 0 .. S - 1 of its slot, with S = max(starts) + T. Columns past a row's own
           length hold its pads or stale data; the attention mask hides them.

        Example: slot_ids = [0, 2], starts = [5, 3], T = 1
            writes column 5 of slot 0 and column 3 of slot 2, returns columns 0-5 of both.
        """
        num_new_tokens = k_new.shape[1]
        cols = rearrange(starts, "b -> b 1") + torch.arange(num_new_tokens)  # [B, T]
        slots = rearrange(slot_ids, "b -> b 1")  # [B, 1], broadcasts against cols
        self.k_cache[layer_idx, slots, cols] = k_new
        self.v_cache[layer_idx, slots, cols] = v_new

        seq_len = int(starts.max()) + num_new_tokens
        k_all = self.k_cache[layer_idx, slot_ids, :seq_len]
        v_all = self.v_cache[layer_idx, slot_ids, :seq_len]
        return k_all, v_all

    @property
    def reserved_bytes(self) -> int:
        """Total bytes pre-allocated in RAM for num_slots * max_seq_len."""
        return (
            self.k_cache.numel() + self.v_cache.numel()
        ) * self.k_cache.element_size()
