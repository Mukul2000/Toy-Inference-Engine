from dataclasses import dataclass, field


@dataclass
class RequestOutput:
    """
    What one request got back. Times are measured from its `add_request()` (with `generate()`,
    the start of the run), so they include any time it spent waiting for a slot.
    """

    token_ids: list[int] = field(default_factory=list)  # generated (incl. EOS if emitted)
    ttft_ms: float = 0.0  # Time To First Token
    latency_ms: float = 0.0  # time until its last token
