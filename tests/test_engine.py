"""
The real model end to end (design_doc.md §6.12, items 1 and 5). Loads SmolLM2 once; ~30 s.
"""

import pytest
import torch

from inference_engine.contract import ScheduledSeq, SchedulerOutput
from inference_engine.engine import BatchingPolicy, Engine
from inference_engine.request import Request

# Batching may flip a token only at a near-tie (§5.4): batch-shape-dependent rounding moves
# logits by ~1e-5..5e-4, so a flip needs the top-2 gap to be about that small.
NEAR_TIE = 1e-2


def assert_matches_reference(model, reference: dict, outputs: list[list[int]]) -> None:
    for item, tokens in zip(reference["requests"], outputs, strict=True):
        ref = item["token_ids"]
        if tokens == ref:
            continue
        k = next((i for i, (x, y) in enumerate(zip(tokens, ref)) if x != y), min(len(tokens), len(ref)))
        top2 = model(torch.tensor([item["prompt_ids"] + ref[:k]]))[0].topk(2).values
        gap = float(top2[0] - top2[1])
        assert gap < NEAR_TIE, f"{item['question']!r} diverged at token {k}, top-2 gap {gap:.4f}"


def test_executor_driven_by_hand_reproduces_reference(smollm_executor, reference):
    """Phase 1 acceptance: one prefill of all 8 prompts, then decodes, with no engine."""
    items = reference["requests"]
    eos, max_new = smollm_executor.config.eos_token_id, reference["max_new_tokens"]
    smollm_executor.allocate_kv_cache(num_slots=len(items), max_seq_len=128)

    outputs = [[] for _ in items]
    rows = [ScheduledSeq(i, i, 0, item["prompt_ids"]) for i, item in enumerate(items)]
    while rows:
        for row, token in zip(rows, smollm_executor.execute(SchedulerOutput(rows))):
            outputs[row.seq_id].append(token)
        rows = [
            ScheduledSeq(row.seq_id, row.slot, row.start + len(row.token_ids), [outputs[row.seq_id][-1]])
            for row in rows
            if outputs[row.seq_id][-1] != eos and len(outputs[row.seq_id]) < max_new
        ]
    assert_matches_reference(smollm_executor.model, reference, outputs)


@pytest.mark.parametrize(
    "policy, num_slots, token_budget",
    [
        (BatchingPolicy.STATIC, 1, None),  # naive
        (BatchingPolicy.STATIC, 3, None),
        (BatchingPolicy.CONTINUOUS, 3, None),  # 8 requests on 3 slots: slots are reused mid-run
        (BatchingPolicy.CONTINUOUS, 3, 16),  # chunked prefill: every prompt is split
    ],
)
def test_every_policy_reproduces_reference(smollm_executor, reference, policy, num_slots, token_budget):
    engine = Engine(smollm_executor, policy, num_slots, max_seq_len=128, token_budget=token_budget)
    requests = [Request(item["prompt_ids"], reference["max_new_tokens"]) for item in reference["requests"]]
    outputs = engine.generate(requests)
    assert_matches_reference(smollm_executor.model, reference, [o.token_ids for o in outputs])


def test_hf_parity_through_meta_none(smollm_executor, reference):
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(reference["model"], dtype=torch.float32).eval()
    prompt = torch.tensor([reference["requests"][-1]["prompt_ids"]])
    with torch.inference_mode():
        expected = hf_model(prompt).logits[:, -1]
    actual = smollm_executor.model(prompt)  # meta=None: one fresh, unpadded row
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
