"""
Layer 2 alone (design_doc.md §6.12 item 3, §7.10 item 3): a tiny random-weight model, driven
by hand. Runs in well under a second.
"""

import pytest
import torch

from inference_engine.config import ModelConfig
from inference_engine.contract import ScheduledSeq, SchedulerOutput
from inference_engine.executor import BatchMetadata, ContiguousKVCache, ModelExecutor
from inference_engine.models.transformer import TransformerModel

ATOL = 1e-5


@pytest.fixture(scope="module")
def model() -> TransformerModel:
    return TransformerModel(ModelConfig.tiny())  # random weights (seed 42)


def new_executor(model: TransformerModel, fill: str = "zeros") -> ModelExecutor:
    executor = ModelExecutor(model)
    executor.allocate_kv_cache(num_slots=4, max_seq_len=32)
    if fill == "random":  # stale data from imaginary previous owners
        executor.kv_cache.k_cache.normal_()
        executor.kv_cache.v_cache.normal_()
    return executor


def run_rows(executor: ModelExecutor, rows: list[tuple[int, int, list[int]]]) -> torch.Tensor:
    """rows = [(slot, start, token_ids)], in any order -> logits [B, V], in the same order."""
    return executor.logits(SchedulerOutput([ScheduledSeq(i, *row) for i, row in enumerate(rows)]))


def one_pass(model: TransformerModel, prompt: list[int]) -> torch.Tensor:
    """Reference: the whole prompt as one fresh row, no cache. Logits [V]."""
    return model(torch.tensor([prompt]))[0]


A = [11, 12, 13, 14, 15, 16]
B = [21, 22, 23, 24, 25, 26, 27, 28, 29]


def test_batch_metadata_matches_the_design_doc_example():
    # §7.7, step k+1: R4's chunk (slot 3, columns 5-9), then R1-R3's decodes
    meta = BatchMetadata.build(slot_ids=[3, 0, 1, 2], starts=[5, 8, 5, 6], lengths=[5, 1, 1, 1])
    assert meta.positions.tolist() == [5, 6, 7, 8, 9, 8, 5, 6]
    assert meta.slots.tolist() == [3, 3, 3, 3, 3, 0, 1, 2]
    assert meta.last_idx.tolist() == [4, 5, 6, 7]
    chunk, decode = meta.groups
    assert (chunk.tokens, chunk.num_rows, chunk.width, chunk.seq_len) == (slice(0, 5), 1, 5, 10)
    assert chunk.attn_mask[0, 0, 0].tolist() == [True] * 6 + [False] * 4  # p5 sees columns 0-5
    assert (decode.tokens, decode.num_rows, decode.width, decode.seq_len) == (slice(5, 8), 3, 1, 9)
    assert decode.attn_mask[1, 0, 0].tolist() == [True] * 6 + [False] * 3  # b' at column 5


def test_multi_token_rows_must_come_first():
    with pytest.raises(ValueError):
        BatchMetadata.build(slot_ids=[0, 1], starts=[8, 0], lengths=[1, 5])


def test_a_rows_output_does_not_depend_on_its_slot_or_batchmates(model):
    alone = run_rows(new_executor(model), [(0, 0, A)])
    batched = run_rows(new_executor(model), [(0, 0, B), (2, 0, A)])  # A in slot 2, beside B
    torch.testing.assert_close(batched[1], alone[0], atol=ATOL, rtol=0)


@pytest.mark.parametrize("chunk", [1, 2, 4, 9])
def test_chunked_prefill_matches_one_pass(model, chunk):
    executor = new_executor(model)
    for start in range(0, len(B), chunk):  # B in chunks of `chunk` tokens, all in slot 3
        logits = run_rows(executor, [(3, start, B[start : start + chunk])])
    torch.testing.assert_close(logits[0], one_pass(model, B), atol=ATOL, rtol=0)


def test_mixed_step_chunks_and_decodes(model):
    """One pass holding: an unfinished prefill's next chunk, a new prompt, and a decode row."""
    executor = new_executor(model)
    run_rows(executor, [(0, 0, B[:4]), (2, 0, A[:-1])])  # B's first chunk; A's prompt
    logits = run_rows(
        executor,
        [
            (2, len(A) - 1, [A[-1]]),  # decode row, listed first: the executor must reorder
            (0, 4, B[4:]),  # B's second chunk, from column 4
            (1, 0, A[:3]),  # a new prompt in slot 1, from column 0
        ],
    )
    torch.testing.assert_close(logits[0], one_pass(model, A), atol=ATOL, rtol=0)
    torch.testing.assert_close(logits[1], one_pass(model, B), atol=ATOL, rtol=0)
    torch.testing.assert_close(logits[2], one_pass(model, A[:3]), atol=ATOL, rtol=0)


def test_stale_data_is_never_read(model):
    def serve(executor):
        run_rows(executor, [(2, 0, A[:-1]), (0, 0, B[:-1])])
        return run_rows(executor, [(2, len(A) - 1, [A[-1]]), (0, len(B) - 1, [B[-1]])])

    torch.testing.assert_close(serve(new_executor(model, "random")), serve(new_executor(model)), atol=ATOL, rtol=0)


def test_execute_returns_one_greedy_token_per_row(model):
    executor = new_executor(model)
    batch = SchedulerOutput([ScheduledSeq(0, 1, 0, A), ScheduledSeq(1, 3, 0, B)])
    assert executor.execute(batch) == [int(one_pass(model, p).argmax()) for p in (A, B)]
