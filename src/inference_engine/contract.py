"""
Everything that crosses between Layer 1 (`engine/`, deciding) and Layer 2 (`executor/`,
computing). See design_doc.md §6.7.

Plain Python only, no torch: Layer 1 imports this module, and Layer 1 must run without torch.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Boot, Layer 2 -> Layer 1: the model's limits that Layer 1 must respect."""

    max_model_len: int  # 8192 for SmolLM2, the RoPE table: no slot may be longer
    eos_token_id: int  # 2 for SmolLM2, for the stop check


@dataclass(frozen=True)
class ScheduledSeq:
    """
    One row of a step: "run these tokens, starting at column `start` of slot `slot`".

    A prefill row starts at 0 and carries the prompt. A decode row starts at the sequence's
    length and carries the token sampled in the previous step.
    """

    seq_id: int
    slot: int  # whose KV to read and extend
    start: int  # column (= position) of token_ids[0] = tokens already in the slot
    token_ids: list[int]  # the tokens whose K/V is not in the slot yet


@dataclass(frozen=True)
class SchedulerOutput:
    """Every step, Layer 1 -> Layer 2: the rows to run."""

    seqs: list[ScheduledSeq]


class Executor(ABC):
    """Everything Layer 1 knows about Layer 2."""

    @abstractmethod
    def model_spec(self) -> ModelSpec:
        """Boot: the model's limits."""

    @abstractmethod
    def allocate_kv_cache(self, num_slots: int, max_seq_len: int) -> None:
        """Boot: reserve `num_slots` slots of `max_seq_len` columns each."""

    @abstractmethod
    def execute(self, batch: SchedulerOutput) -> list[int]:
        """Every step: run the rows and return one sampled token id per row, in row order."""
