from dataclasses import dataclass

import torch
from einops import rearrange


@dataclass(frozen=True)
class AttentionGroup:
    """
    One rectangle of attention inside a packed batch (design_doc.md §7.6, option C2).

    Everything outside attention runs on the packed tokens [N, ...]. Attention regroups one
    contiguous slice of them into a [num_rows, width] rectangle, pads only there, runs one
    SDPA call, and packs the outputs back. A batch has at most two groups: the chunk group
    (rows with more than one token) and the decode group (rows with one token).

    Example: two chunk rows, slot 3 with 5 tokens from column 5, slot 1 with 2 tokens from 0
        tokens   = slice(0, 7)          packed tokens p5 p6 p7 p8 p9 q0 q1
        rect_idx = [0 1 2 3 4 5 6]      where each sits in the flattened [2, 5] rectangle:
                 → [0 1 2 3 4 | 5 6]        row 0: p5 p6 p7 p8 p9
                                            row 1: q0 q1 _  _  _    (pads, inside attention only)
        seq_len  = max(5 + 5, 0 + 2) = 10 key columns read from each slot
    """

    tokens: slice  # this group's tokens in the packed batch
    slot_ids: torch.Tensor  # [num_rows]
    rect_idx: torch.Tensor  # [num_tokens] position of each token in the flattened rectangle
    num_rows: int
    width: int  # the group's longest row
    attn_mask: torch.Tensor  # [num_rows, 1, width, seq_len], True = may attend

    @property
    def seq_len(self) -> int:
        """Key columns read from each slot: the furthest real token of any row, plus one."""
        return self.attn_mask.shape[-1]

    def to_rect(self, x: torch.Tensor) -> torch.Tensor:
        """[num_tokens, ...] -> [num_rows, width, ...], pads filled with zeros."""
        rect = x.new_zeros(self.num_rows * self.width, *x.shape[1:])
        rect[self.rect_idx] = x
        return rect.view(self.num_rows, self.width, *x.shape[1:])

    def from_rect(self, rect: torch.Tensor) -> torch.Tensor:
        """[num_rows, width, ...] -> [num_tokens, ...], pads dropped."""
        return rect.reshape(self.num_rows * self.width, *rect.shape[2:])[self.rect_idx]

    @classmethod
    def build(
        cls, offset: int, slot_ids: list[int], starts: list[int], lengths: list[int]
    ) -> "AttentionGroup":
        width = max(lengths)
        rect_idx = torch.cat([b * width + torch.arange(n) for b, n in enumerate(lengths)])
        seq_len = max(s + n for s, n in zip(starts, lengths))

        q_col = rearrange(torch.tensor(starts), "b -> b 1") + torch.arange(width)  # [rows, width]
        k_col = torch.arange(seq_len)
        # One rule: attend iff k_col <= q_col. A row's real tokens come before its pads and
        # before stale columns of its slot, so this hides future tokens, pads and stale data.
        attn_mask = rearrange(k_col, "s -> 1 1 s") <= rearrange(q_col, "b t -> b t 1")

        return cls(
            tokens=slice(offset, offset + sum(lengths)),
            slot_ids=torch.tensor(slot_ids),
            rect_idx=rect_idx,
            num_rows=len(lengths),
            width=width,
            attn_mask=rearrange(attn_mask, "b t s -> b 1 t s"),
        )
