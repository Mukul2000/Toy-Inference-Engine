import heapq
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
    Layer 1's decisions (design_doc.md §6.5): who waits, who runs this step, which slot each
    running sequence owns, and when a sequence is done. Plain integers only: it knows nothing
    about tensors or the model.
    """

    def __init__(self, num_slots: int, policy: BatchingPolicy, eos_token_id: int):
        self.policy = policy
        self.eos_token_id = eos_token_id
        self.waiting: deque[Sequence] = deque()
        self.running: dict[int, Sequence] = {}  # seq_id -> sequence, in admission order
        self.free_slots: list[int] = list(range(num_slots))  # min-heap: lowest free slot first

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput | None:
        """
        Prefill-first: if the policy allows admission and a slot is free, admit waiting
        sequences in arrival order (one per free slot) and prefill them in one pass; running
        sequences sit this step out. Otherwise decode one token for every running sequence.
        """
        may_admit = self.policy is BatchingPolicy.CONTINUOUS or not self.running
        if may_admit and self.waiting and self.free_slots:
            rows = []
            while self.waiting and self.free_slots:
                rows.append(self._admit(self.waiting.popleft()))
        else:
            rows = list(self.running.values())
        if not rows:
            return None
        return SchedulerOutput(
            [ScheduledSeq(seq.seq_id, seq.slot, seq.num_computed_tokens, seq.next_input()) for seq in rows]
        )

    def update(self, batch: SchedulerOutput, token_ids: list[int], now: float) -> list[Sequence]:
        """Appends each row's sampled token; finishes sequences on EOS or max_new_tokens and frees their slots."""
        finished = []
        for row, token_id in zip(batch.seqs, token_ids, strict=True):
            seq = self.running[row.seq_id]
            seq.num_computed_tokens += len(row.token_ids)
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
