"""Layer 1 must run without torch (design_doc.md §6.12, item 4). Checked statically."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "inference_engine"
LAYER_1 = sorted((PACKAGE / "engine").glob("*.py")) + [PACKAGE / "contract.py", PACKAGE / "request.py", PACKAGE / "metrics.py"]
FORBIDDEN = {"torch", "einops", "transformers", "numpy", "inference_engine.executor", "inference_engine.models"}


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", LAYER_1, ids=lambda p: p.name)
def test_layer_1_imports_no_torch_and_no_layer_2(path):
    bad = {m for m in imported_modules(path) if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
    assert not bad, f"{path.name} imports {bad}"
