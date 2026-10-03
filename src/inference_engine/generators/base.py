from abc import ABC, abstractmethod

from inference_engine.metrics import GenerationResult
from inference_engine.models.transformer import TransformerModel
from inference_engine.request import Request


class Generator(ABC):
    """
    Common interface for every generation strategy: naive, static batching, continuous batching...

    Contract: all `requests` are waiting when `generate()` starts. Each one is generated greedily
    until it emits EOS or reaches its own `max_new_tokens`. Returns one output per request, in
    request order. HOW requests are grouped into forward passes (the scheduling policy) is up to
    each subclass, and that is exactly what we compare across strategies.
    """

    def __init__(self, model: TransformerModel):
        self.model = model
        self.config = model.config

    @abstractmethod
    def generate(self, requests: list[Request]) -> GenerationResult:
        """Generate completions for all `requests`."""
