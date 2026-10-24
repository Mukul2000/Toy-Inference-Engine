import torch

from inference_engine.config import DEFAULT_MODEL_NAME
from inference_engine.contract import Executor, ModelSpec, SchedulerOutput
from inference_engine.executor.batch_metadata import BatchMetadata
from inference_engine.executor.kv_cache import ContiguousKVCache
from inference_engine.models.transformer import TransformerModel


class ModelExecutor(Executor):
    """
    Layer 2 (design_doc.md §6.6, §7.7): runs whatever rows it is handed, samples one token
    per row, and owns the KV slot pool. It never asks whether a step is prefill or decode:
    rows are grouped by shape (one token or more), not by kind.
    """

    def __init__(self, model: TransformerModel):
        self.model = model
        self.config = model.config
        self.dtype = next(model.parameters()).dtype
        self.kv_cache: ContiguousKVCache | None = None

    @classmethod
    def from_pretrained(
        cls, model_name: str = DEFAULT_MODEL_NAME, dtype: torch.dtype = torch.float32
    ) -> "ModelExecutor":
        return cls(TransformerModel.from_pretrained(model_name, dtype=dtype))

    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            max_model_len=self.config.max_position_embeddings,
            eos_token_id=self.config.eos_token_id,
        )

    def allocate_kv_cache(self, num_slots: int, max_seq_len: int) -> None:
        self.kv_cache = ContiguousKVCache(self.config, num_slots, max_seq_len, dtype=self.dtype)

    def execute(self, batch: SchedulerOutput) -> list[int]:
        return self.logits(batch).argmax(dim=-1).tolist()  # greedy sampling

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
