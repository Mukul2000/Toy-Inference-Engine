from dataclasses import dataclass


@dataclass
class StepMetric:
    """Telemetry recorded for a single autoregressive step."""

    step_idx: int
    num_input_tokens: int
    token_id: int
    step_latency_ms: float
    kv_cache_bytes: int


@dataclass
class GenerationResult:
    """Complete result and profiling metrics for a generation run."""

    token_ids: list[int]
    total_time_ms: float
    tokens_per_second: float
    step_metrics: list[StepMetric]

    @property
    def ttft_ms(self) -> float:
        """Time To First Token (Step 0 latency)."""
        return self.step_metrics[0].step_latency_ms if self.step_metrics else 0.0

    @property
    def avg_itl_ms(self) -> float:
        """Average Inter-Token Latency across decode steps (Steps 1..N)."""
        if len(self.step_metrics) <= 1:
            return 0.0
        decode_latencies = [m.step_latency_ms for m in self.step_metrics[1:]]
        return sum(decode_latencies) / len(decode_latencies)
