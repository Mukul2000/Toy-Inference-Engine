from inference_engine.generators.static_batch import StaticBatchGenerator
from inference_engine.models.transformer import TransformerModel


class NaiveGenerator(StaticBatchGenerator):
    """
    Milestone 1 baseline: requests run one at a time, with or without the KV cache.

    That is exactly static batching with B = 1 (no padding, nothing to batch), so instead
    of keeping a second copy of the generation loop it reuses StaticBatchGenerator's.
    """

    def __init__(self, model: TransformerModel, use_cache: bool = True):
        super().__init__(model, max_batch_size=1, use_cache=use_cache)
