from dataclasses import dataclass

import torch

from inference_engine.executor.attention_group import AttentionGroup


@dataclass(frozen=True)
class BatchMetadata:
    """
    The tensor layout of ONE forward pass over a packed batch. The executor builds it from the rows Layer 1
    scheduled; every layer passes it down to attention.

    The rows' tokens are laid end to end with no pads: [N] tokens in total. Outside attention
    every op works token by token and only needs `positions`. Attention also needs to know
    which request each token belongs to; it gets that from `groups`.

    Rows must arrive in packed order: every multi-token row (prefill or chunk) before every
    1-token row (decode). Example: R4's chunk, then R1-R3's decodes
        slot_ids = [3, 0, 1, 2]   starts = [5, 8, 5, 6]   lengths = [5, 1, 1, 1]
        packed:     p5 p6 p7 p8 p9 | a' b' c'
        positions = [5  6  7  8  9 | 8  5  6]
        slots     = [3  3  3  3  3 | 0  1  2]
        groups    = chunk group (tokens 0-4, 1 row), decode group (tokens 5-7, 3 rows)
        last_idx  = [4, 5, 6, 7]
    """

    positions: torch.Tensor  # [N] column (= RoPE position) of every token in its slot
    slots: torch.Tensor  # [N] slot of every token: where its K/V is written
    groups: tuple[AttentionGroup, ...]  # chunk group, then decode group (either may be absent)
    last_idx: torch.Tensor  # [B] packed index of each row's last token: it predicts the next one

    @classmethod
    def build(cls, slot_ids: list[int], starts: list[int], lengths: list[int]) -> "BatchMetadata":
        num_chunk_rows = sum(n > 1 for n in lengths)
        if any(n > 1 for n in lengths[num_chunk_rows:]):
            raise ValueError("rows with more than one token must come before 1-token rows")

        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths)])
        slots = torch.repeat_interleave(torch.tensor(slot_ids), torch.tensor(lengths))
        last_idx = torch.tensor(lengths).cumsum(0) - 1

        # Group by shape, not by kind: a 1-token final chunk is a decode row mathematically
        groups, offset = [], 0
        for rows in (range(num_chunk_rows), range(num_chunk_rows, len(lengths))):
            if rows:
                group = AttentionGroup.build(
                    offset,
                    [slot_ids[i] for i in rows],
                    [starts[i] for i in rows],
                    [lengths[i] for i in rows],
                )
                groups.append(group)
                offset = group.tokens.stop

        return cls(positions=positions, slots=slots, groups=tuple(groups), last_idx=last_idx)
