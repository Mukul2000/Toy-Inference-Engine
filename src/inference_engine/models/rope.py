import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self, head_dim: int, max_position_embeddings: int, base: float = 100000.0
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [num_tokens, num_q_heads, head_dim]
            k: [num_tokens, num_kv_heads, head_dim]
            positions: [num_tokens] integer position indices
        """
        cos = self.cos_cached[positions].unsqueeze(1).to(q.dtype)
        sin = self.sin_cached[positions].unsqueeze(1).to(q.dtype)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
