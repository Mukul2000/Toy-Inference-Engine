from dataclasses import dataclass

import torch
from einops import rearrange


@dataclass(frozen=True)
class BatchMetadata:
    """
    The tensor layout of ONE forward pass (design_doc.md §6.9). The executor builds it from
    the rows Layer 1 scheduled; every layer passes it down to attention.

    Rows are right-padded to T = the longest row. Row b's real tokens sit at columns
    starts[b] .. starts[b] + lengths[b] - 1 of its slot, so column = position, and its pads
    land after them, where nothing reads them.

    Example (§6.9, step 1): seq 0 in slot 0 with 5 tokens cached, seq 1 in slot 2 with 3,
    one new token each:
        slot_ids = [0, 2], starts = [5, 3], lengths = [1, 1]
        positions = [[5], [3]]     S = max(starts) + T = 6     last_idx = [0, 0]
        attn_mask, seq 1 (query at column 3):   key column  0 1 2 3 4 5
                                                            ✓ ✓ ✓ ✓ · ·
    """

    positions: torch.Tensor  # [B, T]        column (= RoPE position) of every new token
    attn_mask: torch.Tensor  # [B, 1, T, S]  True = may attend; S = max(starts) + T
    slot_ids: torch.Tensor  # [B]           the slot each row reads and extends
    starts: torch.Tensor  # [B]           column of each row's first new token
    last_idx: torch.Tensor  # [B]           lengths - 1: the token whose output predicts the next one

    @classmethod
    def build(cls, slot_ids: list[int], starts: list[int], lengths: list[int]) -> "BatchMetadata":
        slot_ids_t, starts_t, lengths_t = (
            torch.tensor(v, dtype=torch.long) for v in (slot_ids, starts, lengths)
        )
        num_new_tokens = int(lengths_t.max())
        seq_len = int(starts_t.max()) + num_new_tokens

        q_col = rearrange(starts_t, "b -> b 1") + torch.arange(num_new_tokens)  # [B, T]
        k_col = torch.arange(seq_len)  # [S]
        # One rule: attend iff k_col <= q_col. A row's real tokens come before its pads and
        # before any stale columns of its slot's previous owner, so this one comparison hides
        # future tokens, pads and stale data. Every query sees at least itself: no NaN rows.
        attn_mask = rearrange(k_col, "s -> 1 1 s") <= rearrange(q_col, "b t -> b t 1")

        return cls(
            positions=q_col,
            attn_mask=rearrange(attn_mask, "b t s -> b 1 t s"),
            slot_ids=slot_ids_t,
            starts=starts_t,
            last_idx=lengths_t - 1,
        )
