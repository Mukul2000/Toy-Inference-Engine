from dataclasses import dataclass

import torch

from inference_engine.config import ModelConfig
from inference_engine.models.transformer import TransformerModel


@dataclass(frozen=True)
class ModelFactory:
    """
    A recipe for a model, not the model itself. Under tensor parallelism every rank builds
    (and slices) its own model in its own process, and a built model is too big to ship
    between processes, so the executor hands each rank this small picklable recipe instead.
    """

    config: ModelConfig
    checkpoint: str | None = None  # Hugging Face name; None = random weights (tests)
    dtype: torch.dtype = torch.float32

    @classmethod
    def pretrained(cls, model_name: str, dtype: torch.dtype = torch.float32) -> "ModelFactory":
        # Reads config.json only, so tp can be validated before any process or weight exists
        return cls(ModelConfig.from_pretrained(model_name), model_name, dtype)

    def create_model(self) -> TransformerModel:
        return self.create_sliced_model(rank=0, tp=1)

    def create_sliced_model(self, rank: int, tp: int) -> TransformerModel:
        if self.checkpoint is None:
            return TransformerModel(self.config, rank, tp).to(dtype=self.dtype)
        return TransformerModel.from_pretrained(self.checkpoint, self.dtype, rank, tp)