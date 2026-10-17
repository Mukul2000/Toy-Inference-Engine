import torch
from torch.nn.utils.rnn import pad_sequence

from inference_engine.config import DEFAULT_MODEL_NAME
from inference_engine.contract import Executor, ModelSpec, SchedulerOutput
from inference_engine.executor.batch_metadata import BatchMetadata
from inference_engine.executor.kv_cache import ContiguousKVCache
from inference_engine.models.transformer import TransformerModel


class ModelExecutor(Executor):
    """
    Layer 2 (design_doc.md §6.6): runs whatever rows it is handed, samples one token per row,
    and owns the KV slot pool. It never asks whether a step is prefill or decode.
    """

    def __init__(self, model: TransformerModel):
        self.model = model
        self.config = model.config
        self.dtype = next(model.parameters()).dtype
        # Any id works for padding: pads sit after a row's real tokens, where nothing reads them.
        self.pad_token_id = self.config.eos_token_id
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
        rows = batch.seqs
        input_ids = pad_sequence(  # [B, T], right-padded
            [torch.tensor(row.token_ids, dtype=torch.long) for row in rows],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        meta = BatchMetadata.build(
            slot_ids=[row.slot for row in rows],
            starts=[row.start for row in rows],
            lengths=[len(row.token_ids) for row in rows],
        )
        logits = self.model(input_ids, meta, self.kv_cache)  # [B, V]
        return logits.argmax(dim=-1).tolist()  # greedy sampling
