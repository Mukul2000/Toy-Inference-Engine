import torch
import torch.nn as nn
from einops import rearrange


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
            q: [B, T, num_q_heads, head_dim]
            k: [B, T, num_kv_heads, head_dim]
            positions: [B, T] position of each token within its OWN request
        """
        # Every head of a token gets the same rotation: a heads axis of size 1 broadcasts
        cos = rearrange(self.cos_cached[positions], "b t d -> b t 1 d").to(q.dtype)
        sin = rearrange(self.sin_cached[positions], "b t d -> b t 1 d").to(q.dtype)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
