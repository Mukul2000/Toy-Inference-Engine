import json
from pathlib import Path

import pytest

REFERENCE = Path(__file__).parent / "data" / "reference_tokens.json"


@pytest.fixture(scope="session")
def reference() -> dict:
    """The answer key (design_doc.md §6.11, Phase 0): the pre-M2 naive generator's tokens."""
    return json.loads(REFERENCE.read_text())


@pytest.fixture(scope="session")
def smollm_executor():
    """The real model, loaded once for the whole test session (a few seconds)."""
    from inference_engine.executor import ModelExecutor

    return ModelExecutor.from_pretrained()
