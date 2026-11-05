import datetime
import os

import torch
import torch.distributed as dist

from inference_engine.contract import SchedulerOutput
from inference_engine.executor.batch_metadata import BatchMetadata
from inference_engine.executor.kv_cache import ContiguousKVCache
from inference_engine.models.model_factory import ModelFactory
from inference_engine.models.transformer import TransformerModel

# How long a rank waits in a collective before giving up. Workers sit in a broadcast between
# commands, so this also bounds how long the driver may stay idle (loading, printing, ...).
TIMEOUT = datetime.timedelta(minutes=10)

class Worker:
    """
    One rank: its model slice and its KV slot pool. Every rank runs the same Worker calls in
    the same order (rank 0 from the Engine via ModelExecutor, the others from rank 0's
    broadcasts in worker_main), so the collectives inside the forward pass always pair up.
    """

    def __init__(self, model: TransformerModel, rank: int, tp: int):
        self.model = model  # sliced model
        self.rank, self.tp = rank, tp
        self.dtype = next(model.parameters()).dtype
        self.kv_cache: ContiguousKVCache | None = None

    def allocate_kv_cache(self, num_slots: int, max_seq_len: int) -> None:
        # TODO: validate the cache dimensions
        self.kv_cache = ContiguousKVCache(self.model.config, num_slots, max_seq_len, dtype=self.dtype)

    def logits(self, batch: SchedulerOutput) -> torch.Tensor:
        """Runs one forward pass. Returns each row's next-token logits [B, V], in row order."""
        rows = batch.seqs
        # Pack multi-token rows first, so each attention group is a contiguous slice
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].token_ids) == 1)
        packed = [rows[i] for i in order]
        input_ids = torch.tensor([t for row in packed for t in row.token_ids])  # [N], no pads
        meta = BatchMetadata.build(
            slot_ids=[row.slot for row in packed],
            starts=[row.start for row in packed],
            lengths=[len(row.token_ids) for row in packed],
        )
        logits = self.model(input_ids, meta, self.kv_cache)  # [B, V], packed order
        return logits[torch.argsort(torch.tensor(order))]  # back to the order rows arrived in


def init_distributed(init_method: str, rank: int, tp: int) -> None:
    """Joins the gloo process group; returns once all tp ranks have joined."""
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=tp, timeout=TIMEOUT)
    torch.set_num_threads(max(1, os.cpu_count() // tp))  # tp processes share the cores


def worker_main(init_method: str, rank: int, tp: int, model_factory: ModelFactory) -> None:
    """Entry point of ranks 1..tp-1: build this rank's Worker, then replay rank 0's calls."""
    init_distributed(init_method, rank, tp)
    worker = Worker(model_factory.create_sliced_model(rank, tp), rank, tp)
    while True:
        msg = [None]
        dist.broadcast_object_list(msg, src=0)
        method, args = msg[0]
        if method == "stop":
            break
        getattr(worker, method)(*args)  # result discarded: rank 0 computes the same one
    dist.destroy_process_group()
