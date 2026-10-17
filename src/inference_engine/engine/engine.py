import time

from inference_engine.contract import Executor, SchedulerOutput
from inference_engine.engine.scheduler import BatchingPolicy, Scheduler
from inference_engine.engine.sequence import Sequence
from inference_engine.metrics import RequestOutput
from inference_engine.request import Request


class Engine:
    """
    Layer 1's front door (design_doc.md §6.10). Constructing it is the boot handshake (§6.7);
    each `step()` is one forward pass: schedule -> execute -> update, as in vLLM's engine core.
    """

    def __init__(
        self,
        executor: Executor,
        policy: BatchingPolicy = BatchingPolicy.CONTINUOUS,
        num_slots: int = 8,
        max_seq_len: int = 512,
    ):
        spec = executor.model_spec()  # up: the model's limits
        if max_seq_len > spec.max_model_len:
            raise ValueError(f"max_seq_len {max_seq_len} > the model's {spec.max_model_len}")
        executor.allocate_kv_cache(num_slots, max_seq_len)  # down: the pool shape Layer 1 chose
        self.executor = executor
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.scheduler = Scheduler(num_slots, policy, spec.eos_token_id)
        # Per step: the rows that ran, and slot -> seq_id of every running sequence (for timelines)
        self.trace: list[tuple[SchedulerOutput, dict[int, int]]] = []
        self._seqs: dict[int, Sequence] = {}
        self._next_seq_id = 0

    def add_request(self, request: Request) -> int:
        """Queues `request` and returns its seq_id. Rejects requests that can't fit a slot."""
        if len(request.prompt_ids) + request.max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({len(request.prompt_ids)}) + max_new_tokens ({request.max_new_tokens}) "
                f"> max_seq_len ({self.max_seq_len})"
            )
        seq = Sequence(self._next_seq_id, request, arrival_time=time.perf_counter())
        self._next_seq_id += 1
        self._seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self) -> list[Sequence]:
        """Runs one forward pass. Returns the sequences that finished in it."""
        batch = self.scheduler.schedule()
        if batch is None:
            return []
        occupancy = {seq.slot: seq.seq_id for seq in self.scheduler.running.values()}
        token_ids = self.executor.execute(batch)
        finished = self.scheduler.update(batch, token_ids, now=time.perf_counter())
        self.trace.append((batch, occupancy))
        return finished

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def generate(self, requests: list[Request]) -> list[RequestOutput]:
        """Offline use: adds every request, steps until all are done, returns outputs in request order."""
        seq_ids = [self.add_request(r) for r in requests]
        while self.has_unfinished():
            self.step()
        return [self._output(self._seqs.pop(seq_id)) for seq_id in seq_ids]

    @staticmethod
    def _output(seq: Sequence) -> RequestOutput:
        return RequestOutput(
            token_ids=seq.output_ids,
            ttft_ms=(seq.first_token_time - seq.arrival_time) * 1000.0,
            latency_ms=(seq.finish_time - seq.arrival_time) * 1000.0,
        )
