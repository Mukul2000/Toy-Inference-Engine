"""
Layer 2 alone (design_doc.md §6.12, item 3): a tiny random-weight model, driven by hand.
Runs in well under a second.
"""

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from inference_engine.config import ModelConfig
from inference_engine.contract import ScheduledSeq, SchedulerOutput
from inference_engine.executor import BatchMetadata, ContiguousKVCache, ModelExecutor
from inference_engine.models.transformer import TransformerModel

ATOL = 1e-5


@pytest.fixture(scope="module")
def model() -> TransformerModel:
    return TransformerModel(ModelConfig.tiny())  # random weights (seed 42)


def new_pool(model: TransformerModel, num_slots: int = 4, fill: str = "zeros") -> ContiguousKVCache:
    pool = ContiguousKVCache(model.config, num_slots, max_seq_len=32)
    if fill == "random":  # stale data from imaginary previous owners
        pool.k_cache.normal_()
        pool.v_cache.normal_()
    return pool


def run_rows(model, pool, rows: list[tuple[int, int, list[int]]]) -> torch.Tensor:
    """rows = [(slot, start, token_ids)] -> logits [B, V], exactly as ModelExecutor.execute builds them."""
    input_ids = pad_sequence([torch.tensor(t) for _, _, t in rows], batch_first=True, padding_value=0)
    meta = BatchMetadata.build([s for s, _, _ in rows], [st for _, st, _ in rows], [len(t) for _, _, t in rows])
    return model(input_ids, meta, pool)


A = [11, 12, 13, 14, 15, 16]
B = [21, 22, 23, 24, 25, 26, 27, 28, 29]


def test_batch_metadata_matches_the_design_doc_example():
    meta = BatchMetadata.build(slot_ids=[0, 2], starts=[5, 3], lengths=[1, 1])
    assert meta.positions.tolist() == [[5], [3]]
    assert meta.last_idx.tolist() == [0, 0]
    assert meta.attn_mask.shape == (2, 1, 1, 6)
    assert meta.attn_mask[1, 0, 0].tolist() == [True, True, True, True, False, False]


def test_a_rows_output_does_not_depend_on_its_slot_or_batchmates(model):
    alone = run_rows(model, new_pool(model), [(0, 0, A)])
    # A in slot 2, right-padded to B's length 9, sharing the pass with B in slot 0
    batched = run_rows(model, new_pool(model), [(0, 0, B), (2, 0, A)])
    torch.testing.assert_close(batched[1], alone[0], atol=ATOL, rtol=0)


def test_prefill_then_decode_matches_one_pass(model):
    one_pass = model(torch.tensor([A]))  # meta=None, no cache
    pool = new_pool(model)
    run_rows(model, pool, [(3, 0, A[:3])])  # prefill 3 tokens into slot 3
    for i in range(3, len(A)):  # then feed the rest one token at a time
        logits = run_rows(model, pool, [(3, i, [A[i]])])
    torch.testing.assert_close(logits, one_pass, atol=ATOL, rtol=0)


def test_decode_rows_at_different_lengths_in_one_pass(model):
    pool = new_pool(model)
    run_rows(model, pool, [(1, 0, A[:-1]), (0, 0, B[:-1])])  # prefill both, right-padded
    # B's prefill was the wider row, so A's slot holds pads at columns 5-7. A's decode
    # overwrites column 5 and must never see columns 6-7.
    logits = run_rows(model, pool, [(1, len(A) - 1, [A[-1]]), (0, len(B) - 1, [B[-1]])])
    torch.testing.assert_close(logits[0], model(torch.tensor([A]))[0], atol=ATOL, rtol=0)
    torch.testing.assert_close(logits[1], model(torch.tensor([B]))[0], atol=ATOL, rtol=0)


def test_stale_data_is_never_read(model):
    def serve(pool):
        run_rows(model, pool, [(2, 0, A[:-1]), (0, 0, B[:-1])])
        return run_rows(model, pool, [(2, len(A) - 1, [A[-1]]), (0, len(B) - 1, [B[-1]])])

    torch.testing.assert_close(serve(new_pool(model, fill="random")), serve(new_pool(model)), atol=ATOL, rtol=0)


def test_execute_returns_one_greedy_token_per_row(model):
    executor = ModelExecutor(model)
    executor.allocate_kv_cache(num_slots=4, max_seq_len=32)
    batch = SchedulerOutput([ScheduledSeq(0, 1, 0, A), ScheduledSeq(1, 3, 0, B)])
    expected = [int(model(torch.tensor([p]))[0].argmax()) for p in (A, B)]
    assert executor.execute(batch) == expected
