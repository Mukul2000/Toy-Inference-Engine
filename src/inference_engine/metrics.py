from dataclasses import dataclass, field


@dataclass
class StepMetric:
    """Telemetry recorded for a single forward pass."""

    step_idx: int  # forward pass index within the whole run
    is_prefill: bool  # True for the first forward pass of a batch
    batch_size: int  # rows in this forward pass
    num_input_tokens: int  # B * T tokens computed (pads and finished rows included)
    num_pad_tokens: int  # how many of those were left padding
    num_active_rows: int  # rows still generating; the other rows are tail waste
    step_latency_ms: float
    kv_cache_bytes: int


@dataclass
class RequestOutput:
    """
    What one request got back. Times are measured from the start of `generate()`,
    so they include any time the request spent waiting behind earlier requests.
    """

    token_ids: list[int] = field(default_factory=list)  # generated (incl. EOS if emitted)
    ttft_ms: float = 0.0  # Time To First Token
    latency_ms: float = 0.0  # time until its last token


@dataclass
class GenerationResult:
    """One output per request (in request order) plus profiling metrics for the whole run."""

    outputs: list[RequestOutput]
    total_time_ms: float
    step_metrics: list[StepMetric]

    @property
    def num_generated_tokens(self) -> int:
        return sum(len(o.token_ids) for o in self.outputs)

    @property
    def tokens_per_second(self) -> float:
        """Useful generated tokens per second, across all requests."""
        return self.num_generated_tokens / (self.total_time_ms / 1000.0)

    @property
    def avg_ttft_ms(self) -> float:
        return sum(o.ttft_ms for o in self.outputs) / len(self.outputs)

    @property
    def avg_latency_ms(self) -> float:
        return sum(o.latency_ms for o in self.outputs) / len(self.outputs)

    @property
    def avg_itl_ms(self) -> float:
        """Average Inter-Token Latency: mean latency of a decode step."""
        decode = [m.step_latency_ms for m in self.step_metrics if not m.is_prefill]
        return sum(decode) / len(decode) if decode else 0.0

    @property
    def padding_waste(self) -> float:
        """Fraction of prefill tokens that were pads."""
        prefill = [m for m in self.step_metrics if m.is_prefill]
        pads = sum(m.num_pad_tokens for m in prefill)
        return pads / sum(m.num_input_tokens for m in prefill)

    @property
    def tail_waste(self) -> float:
        """Fraction of decode row-slots spent on requests that had already finished."""
        decode = [m for m in self.step_metrics if not m.is_prefill]
        if not decode:
            return 0.0
        active = sum(m.num_active_rows for m in decode)
        return 1.0 - active / sum(m.batch_size for m in decode)
