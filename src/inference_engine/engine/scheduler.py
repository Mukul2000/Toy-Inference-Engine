import heapq
import sys
from collections import deque
from enum import Enum

from inference_engine.contract import ScheduledSeq, SchedulerOutput
from inference_engine.engine.sequence import Sequence, SequenceStatus


class BatchingPolicy(Enum):
    """When waiting requests may be admitted (design_doc.md §6.8). Naive = either, with 1 slot."""

    STATIC = "static"  # only when the whole batch has finished
    CONTINUOUS = "continuous"  # whenever a slot is free


class Scheduler:
    """
    Layer 1's decisions (design_doc.md §6.5, §7.3): who waits, who runs this step and with how
    many tokens, which slot each running sequence owns, and when a sequence is done. Plain
    integers only: it knows nothing about tensors or the model.
    """

    def __init__(
        self,
        num_slots: int,
        policy: BatchingPolicy,
        eos_token_id: int,
        token_budget: int | None = None,
    ):
        self.policy = policy
        self.eos_token_id = eos_token_id
        self.token_budget = token_budget  # max tokens per step; None = unlimited (no chunking)
        self.waiting: deque[Sequence] = deque()
        self.running: dict[int, Sequence] = {}  # seq_id -> sequence, in admission order
        self.free_slots: list[int] = list(range(num_slots))  # min-heap: lowest free slot first

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput | None:
        """
        Fills one step's token budget in priority order (§7.3):
          1. one token for every decoding sequence, so nobody stalls;
          2. the rest of the prompt already being prefilled, as much as fits;
          3. newcomers, if the policy allows, in arrival order while budget and slots remain.
        Only the last prompt admitted can be cut short, so at most one prefill is unfinished.
        With no budget, prompts go in whole: static batching then runs as in §6.
        """
        budget = self.token_budget if self.token_budget is not None else sys.maxsize
        may_admit = self.policy is BatchingPolicy.CONTINUOUS or not self.running
        rows: list[ScheduledSeq] = []

        def take(seq: Sequence, limit: int) -> None:  # add one row, spend its tokens
            nonlocal budget
            row = ScheduledSeq(seq.seq_id, seq.slot, seq.num_computed_tokens, seq.next_input(limit))
            rows.append(row)
            budget -= len(row.token_ids)

        for seq in list(self.running.values()):  # 1. every decoding sequence: 1 token
            if not seq.is_prefilling:
                take(seq, 1)
        for seq in list(self.running.values()):  # 2. finish the prefill already in progress
            if seq.is_prefilling and budget > 0:
                take(seq, budget)
        while may_admit and budget > 0 and self.waiting and self.free_slots:  # 3. newcomers
            take(self._admit(self.waiting.popleft()), budget)

        return SchedulerOutput(rows) if rows else None

    def update(self, batch: SchedulerOutput, token_ids: list[int], now: float) -> list[Sequence]:
        """
        Appends each row's sampled token; finishes sequences on EOS or max_new_tokens and frees
        their slots. The sample of a chunk that ends mid-prompt is discarded: it guesses the
        next PROMPT token, which we already know.
        """
        finished = []
        for row, token_id in zip(batch.seqs, token_ids, strict=True):
            seq = self.running[row.seq_id]
            seq.num_computed_tokens += len(row.token_ids)
            if seq.is_prefilling:
                continue
            seq.output_ids.append(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = now
            if token_id == self.eos_token_id or len(seq.output_ids) == seq.request.max_new_tokens:
                self._finish(seq, now)
                finished.append(seq)
        return finished

    def _admit(self, seq: Sequence) -> Sequence:
        seq.slot = heapq.heappop(self.free_slots)
        seq.status = SequenceStatus.RUNNING
        self.running[seq.seq_id] = seq
        return seq

    def _finish(self, seq: Sequence, now: float) -> None:
        # Bookkeeping only: the slot's K/V stays until the next owner overwrites it.
        seq.status = SequenceStatus.FINISHED
        seq.finish_time = now
        del self.running[seq.seq_id]
        heapq.heappush(self.free_slots, seq.slot)
