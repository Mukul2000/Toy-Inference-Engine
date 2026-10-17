import time

import torch
from transformers import AutoTokenizer

from inference_engine.config import DEFAULT_MODEL_NAME
from inference_engine.engine import BatchingPolicy, Engine
from inference_engine.executor import ModelExecutor
from inference_engine.metrics import RequestOutput
from inference_engine.models.transformer import TransformerModel
from inference_engine.request import Request

# Mixed prompt lengths and answer lengths, so slots free up at different steps
QUESTIONS = [
    "What is the capital of France?",
    "What is 12 times 12?",
    "Name three primary colors.",
    "Translate 'good morning' into Spanish.",
    "Write a haiku about the ocean.",
    "Explain in one sentence why the sky is blue.",
    "Give me one short tip for staying productive while working from home.",
    "In large language model inference engines, why is caching the key and value "
    "tensors across autoregressive decoding steps so important for performance?",
]
MAX_NEW_TOKENS = 32
MAX_SEQ_LEN = 128  # per slot: the longest prompt (56) + MAX_NEW_TOKENS fits

# (name, policy, num_slots). The first one is the baseline. 3 slots for 8 requests, so
# slots are freed and reused mid-run (design_doc.md §6.12).
POLICIES = [
    ("Naive (1 slot)", BatchingPolicy.STATIC, 1),
    ("Static (3 slots)", BatchingPolicy.STATIC, 3),
    ("Continuous (3 slots)", BatchingPolicy.CONTINUOUS, 3),
]


def chat_prompt(tokenizer: AutoTokenizer, question: str) -> list[int]:
    """
    Token ids of `question` in the model's chat format, ending with the assistant header
    (`add_generation_prompt=True`) so the model knows it is its turn to answer. Sent as raw
    text, a question looks like a finished user turn: the model usually emits EOS as its very
    first token (an empty answer), or else keeps writing the user's text.
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]


def top2_gap(model: TransformerModel, token_ids: list[int]) -> float:
    """Gap between the best and second-best next-token logit when run alone."""
    top2 = model(torch.tensor([token_ids]))[0].topk(2).values
    return float(top2[0] - top2[1])


def first_mismatch(a: list[int], b: list[int]) -> int:
    """Index of the first token where `a` and `b` differ."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def check_parity(
    tokenizer: AutoTokenizer,
    model: TransformerModel,
    requests: list[Request],
    results: list[tuple[str, list[RequestOutput]]],
) -> None:
    """
    Every policy must reproduce the baseline's tokens. Batching may only flip a token at a
    near-tie, where batch-shape-dependent rounding decides (notes/static_batching.md, section 7).
    """
    base_name, base = results[0]
    for i, out in enumerate(base):
        print(f"[{i}] {tokenizer.decode(out.token_ids)!r}")

    for name, outputs in results[1:]:
        mismatches = 0
        for i, (out, ref) in enumerate(zip(outputs, base)):
            if out.token_ids == ref.token_ids:
                continue
            mismatches += 1
            k = first_mismatch(out.token_ids, ref.token_ids)
            gap = top2_gap(model, requests[i].prompt_ids + ref.token_ids[:k])
            print(
                f"[{i}] {name} differs at token {k} (top-2 logit gap there: {gap:.4f}): "
                f"{tokenizer.decode(out.token_ids)!r}"
            )
        n = len(requests)
        status = "PASSED" if mismatches == 0 else f"{mismatches} request(s) diverged"
        print(f"\n{name} vs. {base_name} parity: {status} ({n - mismatches}/{n} identical)\n")


def print_timeline(name: str, engine: Engine, width: int = 100) -> None:
    """
    The slot timeline of a run, in the format of design_doc.md §6.8, one column per step.
    `kind`: P = prefill step, D = decode step. Each slot shows the request (by index) it holds:
    `-` = holds the slot but paused by a prefill step, `.` = free.
    """
    print(f"{name}: {len(engine.trace)} steps")
    kinds, cells = "", [""] * engine.num_slots
    for batch, occupancy in engine.trace:
        kinds += "P" if batch.seqs[0].start == 0 else "D"
        in_batch = {row.seq_id for row in batch.seqs}
        for slot in range(engine.num_slots):
            seq_id = occupancy.get(slot)
            cells[slot] += "." if seq_id is None else str(seq_id) if seq_id in in_batch else "-"
    for begin in range(0, len(kinds), width):
        end = min(begin + width, len(kinds))
        ruler = "".join(f"{i:<10}" for i in range(begin, end, 10))[: end - begin]  # 0, 10, 20, ...
        print(f"  step      {ruler}")
        print(f"  kind      {kinds[begin:end]}")
        for slot, row in enumerate(cells):
            print(f"  slot {slot}    {row[begin:end]}")
    print()


def print_summary(results: list[tuple[str, list[RequestOutput], float, int]]) -> None:
    """One row per policy. Wall-clock numbers are indicative only: benchmarks come later."""
    print(f"{'Policy':<22} | {'Steps':>5} | {'Total':>8} | {'Throughput':>11} | {'Avg TTFT':>8} | {'Avg latency':>11}")
    print("-" * 82)
    for name, outputs, total_ms, steps in results:
        tokens = sum(len(o.token_ids) for o in outputs)
        ttft = sum(o.ttft_ms for o in outputs) / len(outputs)
        latency = sum(o.latency_ms for o in outputs) / len(outputs)
        print(
            f"{name:<22} | {steps:5d} | {total_ms:5.0f} ms | {tokens / total_ms * 1000:5.1f} tok/s | "
            f"{ttft:5.0f} ms | {latency:8.0f} ms"
        )


def main() -> None:
    """Driver: serves the same requests under every batching policy and compares them."""
    print(f"Loading model & tokenizer: {DEFAULT_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    executor = ModelExecutor.from_pretrained(DEFAULT_MODEL_NAME)

    requests = [Request(chat_prompt(tokenizer, q), MAX_NEW_TOKENS) for q in QUESTIONS]
    lens = [len(r.prompt_ids) for r in requests]
    print(
        f"\n{len(requests)} requests, prompt lengths {min(lens)}..{max(lens)} tokens, "
        f"up to {MAX_NEW_TOKENS} new tokens each\n"
    )

    # Warmup: the first forward passes of a process pay one-time costs (thread-pool start-up,
    # kernel selection, first allocations). A tiny throwaway job keeps them out of whichever
    # policy happens to be measured first.
    warmup = [Request(r.prompt_ids[:4], max_new_tokens=2) for r in requests[:2]]
    Engine(executor, num_slots=2, max_seq_len=MAX_SEQ_LEN).generate(warmup)

    # One executor (one model) serves every policy; each Engine allocates its own slot pool.
    engines, results = [], []
    for name, policy, num_slots in POLICIES:
        engine = Engine(executor, policy, num_slots=num_slots, max_seq_len=MAX_SEQ_LEN)
        t0 = time.perf_counter()
        outputs = engine.generate(requests)
        total_ms = (time.perf_counter() - t0) * 1000.0
        engines.append((name, engine))
        results.append((name, outputs, total_ms, len(engine.trace)))

    check_parity(tokenizer, executor.model, requests, [(name, outputs) for name, outputs, _, _ in results])
    for name, engine in engines[1:]:  # the baseline runs one request at a time: too many steps
        print_timeline(name, engine)
    print_summary(results)


if __name__ == "__main__":
    main()
