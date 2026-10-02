import torch
from inference_engine.config import ModelConfig


class ContiguousKVCache:
    """
    Milestone 1 Contiguous KV Cache.
    Acts as the single source of truth for both cached (K, V) tensors
    and the current sequence position offset (`current_seq_len`).
    """

    def __init__(
        self,
        config: ModelConfig,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.current_seq_len = 0

        cache_shape = (
            config.num_hidden_layers,
            max_seq_len,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.k_cache = torch.zeros(cache_shape, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype)

    def get_positions(self, num_new_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Derives the exact position indices for the incoming `num_new_tokens`
        based on how many tokens are already stored in the cache.
        """
        return torch.arange(
            self.current_seq_len,
            self.current_seq_len + num_new_tokens,
            dtype=torch.long,
            device=device,
        )

    def update(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. Writes the newly computed `k_new, v_new` (shape: [num_new_tokens, ...])
           into the cache at `positions`.
        2. Returns the FULL historical `k_all, v_all` (shape: [total_seq_len, ...]).
        """
        self.k_cache[layer_idx, positions] = k_new
        self.v_cache[layer_idx, positions] = v_new

        total_seq_len = int(positions[-1].item()) + 1
        # Advance current_seq_len once the final layer has updated the cache
        if layer_idx == self.config.num_hidden_layers - 1:
            self.current_seq_len = total_seq_len

        k_all = self.k_cache[layer_idx, :total_seq_len]
        v_all = self.v_cache[layer_idx, :total_seq_len]
        return k_all, v_all

    @property
    def active_bytes(self) -> int:
        """Bytes currently occupied by active cached tokens across all layers."""
        dtype_bytes = self.k_cache.element_size()
        return self.current_seq_len * self.config.kv_bytes_per_token(dtype_bytes)

    @property
    def reserved_bytes(self) -> int:
        """Total bytes pre-allocated in RAM for max_seq_len."""
        return (
            self.k_cache.numel() + self.v_cache.numel()
        ) * self.k_cache.element_size()
