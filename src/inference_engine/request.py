from dataclasses import dataclass


@dataclass
class Request:
    """One generation request: the input type of every Generator."""

    prompt_ids: list[int]
    max_new_tokens: int

    def __post_init__(self):
        if not self.prompt_ids or self.max_new_tokens < 1:
            raise ValueError("A Request needs at least 1 prompt token and max_new_tokens >= 1")
