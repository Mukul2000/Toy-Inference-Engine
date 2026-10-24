from dataclasses import dataclass, field
from enum import Enum

from inference_engine.request import Request


class SequenceStatus(Enum):
    WAITING = "waiting"  # added, no slot yet
    RUNNING = "running"  # owns a slot: prefilled, now decoding
    FINISHED = "finished"  # EOS or max_new_tokens; its slot is free again


@dataclass(eq=False)
class Sequence:
    """
    One request's life inside the engine (design_doc.md §6.10):

        WAITING --admitted, gets a slot, prefilled--> RUNNING --EOS or max_new_tokens--> FINISHED
    """

    seq_id: int
    request: Request
    arrival_time: float  # time.perf_counter() at add_request()
    status: SequenceStatus = SequenceStatus.WAITING
    slot: int | None = None  # owned while RUNNING
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0  # tokens whose K/V is in the slot: the sequence's length
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def is_prefilling(self) -> bool:
        """Some prompt tokens are not in the slot yet."""
        return self.num_computed_tokens < len(self.request.prompt_ids)

    def next_input(self, limit: int) -> list[int]:
        """
        The next tokens not yet in the cache, at most `limit` of them: this step's input for the row.
          - prefilling:                         the next chunk of the prompt (start = tokens cached)
          - running, all but the newest token:  the newest token (decode row, start = length)
        """
        start = self.num_computed_tokens
        return (self.request.prompt_ids + self.output_ids)[start : start + limit]
