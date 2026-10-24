import torch

from inference_engine.config import ModelConfig


class ContiguousKVCache:
    """
    The KV slot pool for continuous batching (design_doc.md §6.3).

    One pre-allocated buffer each for K and V:
        [num_layers, num_slots, max_seq_len, num_kv_heads, head_dim]

    A request owns one slot from admission until it finishes. Inside its slot, its tokens
    sit at columns 0, 1, 2, ... in order, so column = position.

    Storage only: the pool keeps no lengths and no cursor. Every call says which slots and
    columns to touch; Layer 1 decides both. A freed slot is never cleared: its next owner
    overwrites it from column 0, and the `k <= q` mask (see `AttentionGroup`) never lets a
    request read past its own length.
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

    def write(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """
        Scatter: writes packed token i's `k_new[i], v_new[i]` ([N, num_kv_heads, head_dim])
        into column positions[i] of slot slots[i]. Only real tokens: packed batches have no pads.

        Example: slots = [3, 3, 0], positions = [5, 6, 8]
            writes columns 5 and 6 of slot 3, and column 8 of slot 0.
        """
        self.k_cache[layer_idx, slots, positions] = k_new
        self.v_cache[layer_idx, slots, positions] = v_new

    def read(
        self, layer_idx: int, slot_ids: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather: returns columns 0 .. seq_len - 1 of each slot in `slot_ids`, as `k_all, v_all`
        ([B, seq_len, num_kv_heads, head_dim]). Columns past a row's own length hold stale
        data; the attention mask hides them.
        """
        return self.k_cache[layer_idx, slot_ids, :seq_len], self.v_cache[layer_idx, slot_ids, :seq_len]

    @property
    def reserved_bytes(self) -> int:
        """Total bytes pre-allocated in RAM for num_slots * max_seq_len."""
        return (
            self.k_cache.numel() + self.v_cache.numel()
        ) * self.k_cache.element_size()
