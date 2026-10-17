"""
Layer 1 alone (design_doc.md §6.12, item 2): the Engine and Scheduler against a fake executor.
No model and no tensors: every request emits token 7 until it reaches max_new_tokens.
"""

import pytest

from inference_engine.contract import Executor, ModelSpec, SchedulerOutput
from inference_engine.engine import BatchingPolicy, Engine
from inference_engine.request import Request

EOS = 2
TOKEN = 7


class FakeExecutor(Executor):
    """Layer 2 stand-in: records every batch; `eos_at[seq_id] = n` makes that sequence's n-th token EOS."""

    def __init__(self, eos_at: dict[int, int] | None = None):
        self.eos_at = eos_at or {}
        self.batches: list[SchedulerOutput] = []
        self.emitted: dict[int, int] = {}

    def model_spec(self) -> ModelSpec:
        return ModelSpec(max_model_len=8192, eos_token_id=EOS)

    def allocate_kv_cache(self, num_slots: int, max_seq_len: int) -> None:
        self.allocated = (num_slots, max_seq_len)

    def execute(self, batch: SchedulerOutput) -> list[int]:
        self.batches.append(batch)
        out = []
        for row in batch.seqs:
            n = self.emitted[row.seq_id] = self.emitted.get(row.seq_id, 0) + 1
            out.append(EOS if self.eos_at.get(row.seq_id) == n else TOKEN)
        return out


def make_requests(new_tokens: list[int]) -> list[Request]:
    """Request i has a prompt of i + 2 tokens (all 100 + i) and needs new_tokens[i] tokens."""
    return [Request([100 + i] * (i + 2), n) for i, n in enumerate(new_tokens)]


def run(policy: BatchingPolicy, num_slots: int, new_tokens: list[int], **kw) -> tuple[Engine, list]:
    engine = Engine(FakeExecutor(**kw), policy, num_slots=num_slots, max_seq_len=64)
    return engine, engine.generate(make_requests(new_tokens))


# The running example of design_doc.md §6.8: R1 needs 6 tokens, R2 and R3 need 2, R4 and R5 need 3.
EXAMPLE = [6, 2, 2, 3, 3]


@pytest.mark.parametrize(
    "policy, num_slots, steps",
    [
        (BatchingPolicy.STATIC, 3, 9),
        (BatchingPolicy.CONTINUOUS, 3, 7),
        (BatchingPolicy.STATIC, 1, 16),  # naive
        (BatchingPolicy.CONTINUOUS, 1, 16),  # naive, either way
    ],
)
def test_step_counts_match_design_doc(policy, num_slots, steps):
    engine, outputs = run(policy, num_slots, EXAMPLE)
    assert len(engine.trace) == steps
    assert [len(o.token_ids) for o in outputs] == EXAMPLE


def test_continuous_timeline_matches_design_doc():
    engine, _ = run(BatchingPolicy.CONTINUOUS, 3, EXAMPLE)
    kinds = ["P" if batch.seqs[0].start == 0 else "D" for batch, _ in engine.trace]
    assert "".join(kinds) == "PDPDDDD"
    step2, occupancy = engine.trace[2]
    # R4, R5 (seq 3, 4) take the slots R2, R3 freed at step 1; R1 (seq 0) is paused
    assert [(row.seq_id, row.slot) for row in step2.seqs] == [(3, 1), (4, 2)]
    assert occupancy == {0: 0, 1: 3, 2: 4}


def test_static_admits_only_into_an_empty_batch():
    engine, _ = run(BatchingPolicy.STATIC, 3, EXAMPLE)
    for batch, occupancy in engine.trace:
        if batch.seqs[0].start == 0:  # a prefill step: nobody else may be running
            assert set(occupancy.values()) == {row.seq_id for row in batch.seqs}


def test_rows_carry_the_tokens_not_yet_in_the_cache():
    engine, outputs = run(BatchingPolicy.CONTINUOUS, 3, EXAMPLE)
    requests = make_requests(EXAMPLE)
    seen: dict[int, int] = {}  # seq_id -> tokens emitted so far
    for batch, _ in engine.trace:
        for row in batch.seqs:
            prompt = requests[row.seq_id].prompt_ids
            n = seen.get(row.seq_id, 0)
            if n == 0:  # prefill row: the whole prompt, from column 0
                assert (row.start, row.token_ids) == (0, prompt)
            else:  # decode row: the newest token, at the sequence's length
                assert row.start == len(prompt) + n - 1
                assert row.token_ids == [outputs[row.seq_id].token_ids[n - 1]]
            seen[row.seq_id] = n + 1


def test_a_slot_never_holds_two_running_sequences():
    engine = Engine(FakeExecutor(), BatchingPolicy.CONTINUOUS, num_slots=3, max_seq_len=64)
    for request in make_requests([5, 1, 4, 2, 2, 6, 1, 3]):
        engine.add_request(request)
    while engine.has_unfinished():
        engine.step()
        slots = [seq.slot for seq in engine.scheduler.running.values()]
        assert len(slots) == len(set(slots))
        assert len(slots) + len(engine.scheduler.free_slots) == 3


def test_admission_is_first_come_first_served():
    engine, _ = run(BatchingPolicy.CONTINUOUS, 2, [3, 1, 4, 1, 5, 9, 2, 6])
    admitted = [row.seq_id for batch, _ in engine.trace for row in batch.seqs if row.start == 0]
    assert admitted == list(range(8))


def test_eos_finishes_early_and_frees_the_slot():
    engine, outputs = run(BatchingPolicy.CONTINUOUS, 1, [5, 5], eos_at={0: 2})
    assert outputs[0].token_ids == [TOKEN, EOS]
    assert len(outputs[1].token_ids) == 5
    assert len(engine.trace) == 2 + 5  # seq 1 gets the slot right after seq 0's EOS


def test_outputs_and_times():
    _, outputs = run(BatchingPolicy.CONTINUOUS, 3, EXAMPLE)
    assert all(0 < o.ttft_ms <= o.latency_ms for o in outputs)


def test_requests_that_cannot_fit_a_slot_are_rejected():
    engine = Engine(FakeExecutor(), num_slots=2, max_seq_len=10)
    engine.add_request(Request([1] * 6, max_new_tokens=4))  # 6 + 4 = 10 fits
    with pytest.raises(ValueError):
        engine.add_request(Request([1] * 6, max_new_tokens=5))
    with pytest.raises(ValueError):
        Engine(FakeExecutor(), max_seq_len=8193)  # longer than the model's 8192


def test_boot_handshake_allocates_the_pool_layer_1_chose():
    executor = FakeExecutor()
    Engine(executor, num_slots=5, max_seq_len=40)
    assert executor.allocated == (5, 40)
