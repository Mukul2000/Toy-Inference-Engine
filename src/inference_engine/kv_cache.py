import torch
from inference_engine.config import ModelConfig


class ContiguousKVCache:
    """
    Contiguous KV Cache with a batch dimension (static batching).

    One pre-allocated buffer each for K and V:
        [num_layers, batch_size, max_seq_len, num_kv_heads, head_dim]

    Indexed by storage COLUMN, not by position: with left padding, row b's token in
    column c sits at position c - pad_lens[b] (see `inference_engine.padding`).

    `current_seq_len` (columns filled so far) is the single source of truth for where the
    next tokens are written. It is ONE integer for the whole batch because left padding
    keeps all rows in lockstep: every row writes the same column at every step.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.current_seq_len = 0

        # It's space set aside before generation starts, not what actually gets used.
        cache_shape = (
            config.num_hidden_layers, # cache for which layer index
            batch_size, # order of the request in the batch (each request gets its own KV cache) 
            max_seq_len, # longest prompt in the batch + largest max_new_tokens in the batch
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. Writes the newly computed `k_new, v_new` (shape: [B, num_new_tokens, ...])
           into columns [current_seq_len, current_seq_len + num_new_tokens) of every row.
        2. Returns the FULL historical `k_all, v_all` (shape: [B, total_seq_len, ...]).
        """
        start = self.current_seq_len
        end = start + k_new.shape[1]
        self.k_cache[layer_idx, :, start:end] = k_new
        self.v_cache[layer_idx, :, start:end] = v_new

        # Advance current_seq_len once the final layer has updated the cache,
        # so every layer of this forward pass writes the same columns.
        if layer_idx == self.config.num_hidden_layers - 1:
            self.current_seq_len = end

        k_all = self.k_cache[layer_idx, :, :end]
        v_all = self.v_cache[layer_idx, :, :end]
        return k_all, v_all

    @property
    def active_bytes(self) -> int:
        """Bytes occupied by filled columns across all rows and layers (pad columns included)."""
        dtype_bytes = self.k_cache.element_size()
        return (
            self.batch_size
            * self.current_seq_len
            * self.config.kv_bytes_per_token(dtype_bytes)
        )

    @property
    def reserved_bytes(self) -> int:
        """Total bytes pre-allocated in RAM for batch_size * max_seq_len."""
        return (
            self.k_cache.numel() + self.v_cache.numel()
        ) * self.k_cache.element_size()
