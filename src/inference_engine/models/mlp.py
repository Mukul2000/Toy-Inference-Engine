import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from inference_engine.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network used in Llama / SmolLM2 / Qwen architectures."""

    def __init__(self, config: ModelConfig, rank: int = 0, tp: int = 1):
        super().__init__()
        self.rank = rank
        self.tp = tp
        rank_size = config.intermediate_size // tp
        lo, hi = rank * rank_size, (rank + 1) * rank_size
        # Build FULL-size layers first: they draw the same random numbers as at tp=1, so with
        # random weights (tests) rank r's slice equals features lo..hi of the tp=1 MLP, and
        # every layer built after this one is identical too. Pretrained weights overwrite them.
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if tp > 1:
            # weight is [out, in]: gate/up keep output rows lo..hi, down keeps input columns lo..hi
            self.gate_proj.weight = nn.Parameter(self.gate_proj.weight[lo:hi].clone())  # [I/tp, H]
            self.up_proj.weight = nn.Parameter(self.up_proj.weight[lo:hi].clone())      # [I/tp, H]
            self.down_proj.weight = nn.Parameter(self.down_proj.weight[:, lo:hi].clone())  # [H, I/tp]
            self.gate_proj.out_features = self.up_proj.out_features = rank_size
            self.down_proj.in_features = rank_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        if self.tp > 1:
            dist.all_reduce(result)
        return result
