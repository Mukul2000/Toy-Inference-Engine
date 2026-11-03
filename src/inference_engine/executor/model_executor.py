import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from inference_engine.config import DEFAULT_MODEL_NAME, ParallelConfig
from inference_engine.contract import Executor, ModelSpec, SchedulerOutput
from inference_engine.executor.kv_cache import ContiguousKVCache
from inference_engine.executor.worker import Worker, init_distributed, worker_main
from inference_engine.models.model_factory import ModelFactory
from inference_engine.models.transformer import TransformerModel


class ModelExecutor(Executor):
    """
    Layer 2 (design_doc.md §6.6, §7.7): runs whatever rows it is handed, samples one token
    per row, and owns the KV slot pool. It never asks whether a step is prefill or decode:
    rows are grouped by shape (one token or more), not by kind.

    Lives in rank 0 and is the only thing Layer 1 talks to. With tp > 1 it spawns ranks
    1..tp-1 and mirrors every Worker call to them (broadcast, then run it locally); with
    tp == 1 it is a pass-through to one Worker and never touches torch.distributed.
    """

    def __init__(self, model_factory: ModelFactory, parallel_config: ParallelConfig = ParallelConfig()):
        self.tp = parallel_config.tp
        parallel_config.validate(model_factory.config)  # before any process or weight exists
        self.procs: list[mp.Process] = []
        if self.tp > 1:
            init_method = f"tcp://127.0.0.1:{free_port()}"
            ctx = mp.get_context("spawn")  # fresh interpreters: nothing shared but the args
            self.procs = [
                # daemon: if this process exits without shutdown(), the workers die with it
                ctx.Process(target=worker_main, args=(init_method, r, self.tp, model_factory), daemon=True)
                for r in range(1, self.tp)
            ]
            for p in self.procs:
                p.start()
            init_distributed(init_method, 0, self.tp)  # blocks until every rank has joined
        self.worker = Worker(model_factory.create_sliced_model(0, self.tp), 0, self.tp)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        dtype: torch.dtype = torch.float32,
        parallel_config: ParallelConfig = ParallelConfig(),
    ) -> "ModelExecutor":
        return cls(ModelFactory.pretrained(model_name, dtype), parallel_config)

    # Rank 0's view, for tests and the driver's parity checks
    @property
    def model(self) -> TransformerModel:
        return self.worker.model

    @property
    def config(self):
        return self.worker.model.config

    @property
    def kv_cache(self) -> ContiguousKVCache | None:
        return self.worker.kv_cache

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            max_model_len=self.config.max_position_embeddings,
            eos_token_id=self.config.eos_token_id,
        )

    def allocate_kv_cache(self, num_slots: int, max_seq_len: int) -> None:
        self._run("allocate_kv_cache", num_slots, max_seq_len)

    def execute(self, batch: SchedulerOutput) -> list[int]:
        return self.logits(batch).argmax(dim=-1).tolist()  # greedy sampling

    def logits(self, batch: SchedulerOutput) -> torch.Tensor:
        """Runs one forward pass on every rank. Returns rank 0's logits [B, V], in row order."""
        return self._run("logits", batch)

    def shutdown(self) -> None:
        """Stops the workers. Safe to call more than once; a no-op at tp == 1."""
        if not self.procs:
            return
        dist.broadcast_object_list([("stop", ())], src=0)
        for p in self.procs:
            p.join()
        dist.destroy_process_group()
        self.procs = []

    def _run(self, method: str, *args):
        """Every rank must make the same calls in the same order: tell the others, then do it."""
        if self.procs:
            dist.broadcast_object_list([(method, args)], src=0)
        return getattr(self.worker, method)(*args)


def free_port() -> int:
    """A port nobody is listening on, so concurrent or back-to-back runs don't collide."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
