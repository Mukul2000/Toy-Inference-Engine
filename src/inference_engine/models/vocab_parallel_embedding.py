import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class VocabParallelEmbedding(nn.Module):
  def __init__(self, vocab_size: int, hidden_size: int, rank: int = 0, tp: int = 1):
    super().__init__()  # required before assigning parameters / submodules
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.rank = rank
    self.tp = tp

    step = vocab_size // tp
    # Draw the FULL random table, then keep this rank's rows. Same seed -> rank r's rows
    # equal rows lo..hi of the tp=1 table, and every rank consumes the same random numbers,
    # so the layers built after this one get identical random weights too (tests).
    # from_pretrained overwrites the values anyway; only the shape matters there.
    full = torch.randn(vocab_size, hidden_size)
    # nn.Parameter registers it, so its state-dict key is "<attribute name>.weight",
    # i.e. "embed_tokens.weight" — the same key the checkpoint uses
    self.weight = nn.Parameter(full[rank * step : (rank + 1) * step].clone())  # [V/tp, H]

  @torch.inference_mode()
  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """[N] token ids -> [N, H] embeddings, identical on every rank."""
    if self.tp == 1:
      # running in non-distributed manner
      return F.embedding(input_ids, self.weight)  # what nn.Embedding does inside

    step = self.vocab_size // self.tp
    lo = self.rank * step
    hi = (self.rank + 1) * step

    # 1. which ids live in my rows?                       [N] bool
    mask = (input_ids >= lo) & (input_ids < hi)
    # 2. shift into my table's numbering, 3. park foreign ids on a valid row (0),
    #    otherwise they'd index out of range or silently wrap around
    local_ids = torch.where(mask, input_ids - lo, 0)    # [N]
    # 4. lookup: foreign positions get row 0 = junk     [N, H]
    rows = F.embedding(local_ids, self.weight)
    # 5. zero the junk rows: [N, H] * [N, 1] broadcasts one 0/1 per row
    rows = rows * mask.unsqueeze(-1)
    # 6. each token's real row came from exactly one rank: summing glues them back
    dist.all_reduce(rows)  # in place
    return rows