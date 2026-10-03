import torch
from transformers import AutoTokenizer

from inference_engine.config import DEFAULT_MODEL_NAME
from inference_engine.generators import Generator, NaiveGenerator, StaticBatchGenerator
from inference_engine.metrics import GenerationResult
from inference_engine.models.transformer import TransformerModel
from inference_engine.request import Request

# Mixed prompt lengths (=> padding waste) and answer lengths (=> tail waste)
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
    results: list[tuple[str, GenerationResult]],
) -> None:
    """
    Every strategy must reproduce the baseline's tokens. Batching may only flip a token at a
    near-tie, where batch-shape-dependent rounding decides (notes/static_batching.md, section 7).
    """
    base_name, base = results[0]
    for i, out in enumerate(base.outputs):
        print(f"[{i}] {tokenizer.decode(out.token_ids)!r}")

    for name, result in results[1:]:
        mismatches = 0
        for i, (out, ref) in enumerate(zip(result.outputs, base.outputs)):
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


def print_steps(name: str, result: GenerationResult) -> None:
    """Per forward pass telemetry (every 4th step, plus the last one)."""
    print("=" * 80)
    print(f"{name}, per step")
    print(
        f"{'Step':>4} | {'Input tokens':>16} | {'Active rows':>11} | {'Latency':>10} | "
        f"{'KV cache':>9}"
    )
    print("-" * 80)
    last_step = len(result.step_metrics) - 1
    for m in result.step_metrics:
        if m.step_idx % 4 == 0 or m.step_idx == last_step:
            per_row = m.num_input_tokens // m.batch_size
            print(
                f"{m.step_idx:4d} | {m.batch_size:3d} x {per_row:3d} = {m.num_input_tokens:4d} | "
                f"{m.num_active_rows:5d}/{m.batch_size:<5d} | {m.step_latency_ms:7.2f} ms | "
                f"{m.kv_cache_bytes / 2**20:6.2f} MB"
            )
    print("=" * 80)


def print_summary(model: TransformerModel, results: list[tuple[str, GenerationResult]]) -> None:
    """One row per strategy, then the waste that batching introduces."""
    _, base = results[0]
    print(
        f"\nSummary ({len(base.outputs)} requests, {base.num_generated_tokens} generated tokens "
        f"by the baseline):"
    )
    print(
        f"{'Strategy':<20} | {'Total':>9} | {'Throughput':>11} | {'Speedup':>7} | "
        f"{'Avg TTFT':>9} | {'Avg latency':>11} | {'Decode step':>11}"
    )
    print("-" * 96)
    for name, r in results:
        print(
            f"{name:<20} | {r.total_time_ms:6.0f} ms | {r.tokens_per_second:5.1f} tok/s | "
            f"{r.tokens_per_second / base.tokens_per_second:6.2f}x | {r.avg_ttft_ms:6.0f} ms | "
            f"{r.avg_latency_ms:8.0f} ms | {r.avg_itl_ms:8.1f} ms"
        )
    print(
        "TTFT and latency are per request and measured from the start of the run, so time spent\n"
        "waiting behind other requests counts. Decode step = average latency of one decode pass."
    )

    kv_mb_per_token = model.config.kv_bytes_per_token() / 2**20
    print("\nWaste (what continuous batching removes):")
    for name, r in results:
        pad_tokens = sum(m.num_pad_tokens for m in r.step_metrics if m.is_prefill)
        print(
            f"  • {name:<20}: padding {r.padding_waste:5.1%} of prefill tokens "
            f"({pad_tokens * kv_mb_per_token:.2f} MB of KV cache) | "
            f"tail {r.tail_waste:5.1%} of decode row-slots"
        )


def main() -> None:
    """Driver: serves the same requests with every generation strategy and compares them."""
    print(f"Loading model & tokenizer: {DEFAULT_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = TransformerModel.from_pretrained(DEFAULT_MODEL_NAME)

    requests = [Request(chat_prompt(tokenizer, q), MAX_NEW_TOKENS) for q in QUESTIONS]
    lens = [len(r.prompt_ids) for r in requests]
    print(
        f"\n{len(requests)} requests, prompt lengths {min(lens)}..{max(lens)} tokens, "
        f"up to {MAX_NEW_TOKENS} new tokens each\n"
    )

    # Every strategy sits behind the same Generator interface. The first one is the baseline.
    batch_size = len(requests)
    generators: list[tuple[str, Generator]] = [
        ("Naive (B=1)", NaiveGenerator(model)),
        (f"Static batch (B={batch_size})", StaticBatchGenerator(model, max_batch_size=batch_size)),
    ]

    # Warmup: the first forward passes of a process pay one-time costs (thread-pool start-up,
    # kernel selection, first allocations): ~20 ms on this CPU, up to seconds on a GPU. A tiny
    # throwaway job keeps them out of whichever strategy happens to be measured first.
    warmup = [Request(r.prompt_ids[:4], max_new_tokens=2) for r in requests[:2]]
    for _, generator in generators:
        generator.generate(warmup)

    results = [(name, generator.generate(requests)) for name, generator in generators]

    check_parity(tokenizer, model, requests, results)
    for name, result in results[1:]:  # the baseline has one batch per request: too many steps
        print_steps(name, result)
    print_summary(model, results)


if __name__ == "__main__":
    main()
