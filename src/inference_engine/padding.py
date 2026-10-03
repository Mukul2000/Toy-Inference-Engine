"""
Left padding for static batching (see notes/static_batching.md, section 4).

Prompts of different lengths are left-padded into one rectangular [B, T] tensor so that
every row ends at the same column, and decode can move all rows in lockstep:

    column:     0  1  2  3 | 4   5   (decode)
    row b=0:    A  B  C  D | X1  X2      pad_lens[0] = 0
    row b=1:    _  _  E  F | Y1  Y2      pad_lens[1] = 2

Everything the model needs to know about the padding follows from `pad_lens` alone.
"""

import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


def left_pad(
    prompts: list[list[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      input_ids: [B, T_max]  prompts left-padded with `pad_token_id`
                             (any id works: pads are masked out of attention).
      pad_lens:  [B]         number of pads at the start of each row.
    """
    input_ids = pad_sequence(
        [torch.tensor(p, dtype=torch.long) for p in prompts],
        batch_first=True,
        padding_value=pad_token_id,
        padding_side="left",
    )
    pad_lens = input_ids.shape[1] - torch.tensor([len(p) for p in prompts])
    return input_ids, pad_lens


def build_positions_and_mask(
    pad_lens: torch.Tensor, start: int, num_new_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derives RoPE positions and the attention mask for `num_new_tokens` new columns that
    begin at storage column `start` (= columns already in the KV cache, 0 without a cache).

    Storage column != position: a row's real tokens only begin after its pads, so
        position = q_col - pad_len     (pads clamp to 0; their value is never used)

    A query at column q_col may attend to a key at column k_col iff
           k_col <= q_col     causal: never look at the future
       and k_col >= pad_len   never look at a pad (pad_len = pad_lens[b] of the query's row)
        or k_col == q_col     NaN guard: a pad query has no other allowed key, and a fully
                              masked softmax row is NaN, which would leak into real rows.

    Requests never see each other: SDPA attends separately within each batch row b.

    Example (row b=1 above, prefill: start=0, num_new_tokens=4), positions[1] = [0, 0, 0, 1]:
                  k0 k1 k2 k3
        q0 (pad)  ✓  .  .  .
        q1 (pad)  .  ✓  .  .
        q2 (E)    .  .  ✓  .
        q3 (F)    .  .  ✓  ✓

    Returns:
      positions: [B, T]         T = num_new_tokens
      attn_mask: [B, 1, T, S]   S = start + T keys, True = may attend (1 broadcasts over heads)
    """
    device = pad_lens.device
    # Put each index on its own axis of a (b, t, s) = (row, query column, key column) grid.
    # Comparing them then broadcasts into the full [B, T, S] table, with no loops.
    pad_len = rearrange(pad_lens, "b -> b 1 1")
    q_col = rearrange(torch.arange(start, start + num_new_tokens, device=device), "t -> 1 t 1")
    k_col = rearrange(torch.arange(start + num_new_tokens, device=device), "s -> 1 1 s")

    positions = rearrange((q_col - pad_len).clamp(min=0), "b t 1 -> b t")
    attn_mask = ((k_col <= q_col) & (k_col >= pad_len)) | (k_col == q_col)  # [B, T, S]
    return positions, rearrange(attn_mask, "b t s -> b 1 t s")
