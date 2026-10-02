from transformers import AutoTokenizer

from inference_engine.config import DEFAULT_MODEL_NAME
from inference_engine.generator import NaiveGenerator
from inference_engine.models.transformer import TransformerModel


def main() -> None:
    """Driver script for Milestone 1: Naive Recomputation vs. Contiguous KV Cache."""
    model_id = DEFAULT_MODEL_NAME
    print(f"Loading model & tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = TransformerModel.from_pretrained(model_id)
    generator = NaiveGenerator(model)

    prompt = (
        "In distributed systems and large language model inference engines, "
        "caching the key and value tensors across autoregressive steps is critical "
        "because without a cache, the hardware must recompute every historical token "
        "from scratch. Consequently, the primary bottleneck shifts from"
    )
    prompt_ids = tokenizer.encode(prompt)
    max_new_tokens = 24

    print(f"\nPrompt ({len(prompt_ids)} tokens): {prompt!r}")
    print(f"Generating {max_new_tokens} tokens...\n")

    # Warmup pass
    _ = generator.generate(prompt_ids[:4], max_new_tokens=2, use_cache=True)

    # 1. Run WITHOUT KV Cache
    res_no_cache = generator.generate(
        prompt_ids, max_new_tokens=max_new_tokens, use_cache=False
    )

    # 2. Run WITH Contiguous KV Cache
    res_with_cache = generator.generate(
        prompt_ids, max_new_tokens=max_new_tokens, use_cache=True
    )

    # Verify exact token-for-token parity
    assert res_no_cache.token_ids == res_with_cache.token_ids, (
        "Parity check failed: token IDs differ!"
    )

    generated_text = tokenizer.decode(res_with_cache.token_ids)
    print(f"Generated Output: {generated_text!r}")
    print("Exact Output Parity: PASSED (100% identical token IDs)\n")

    print("=" * 80)
    print(
        f"{'Step':>4} | {'No-Cache (Tokens -> Latency)':^32} | {'With KV Cache (Tokens -> Latency, RAM)':^36}"
    )
    print("-" * 80)
    for m_no, m_kv in zip(res_no_cache.step_metrics, res_with_cache.step_metrics):
        if m_no.step_idx % 4 == 0 or m_no.step_idx == max_new_tokens - 1:
            kv_kb = m_kv.kv_cache_bytes / 1024.0
            print(
                f"{m_no.step_idx:4d} | "
                f"{m_no.num_input_tokens:4d} tokens -> {m_no.step_latency_ms:7.2f} ms       | "
                f"{m_kv.num_input_tokens:4d} tokens -> {m_kv.step_latency_ms:7.2f} ms ({kv_kb:6.0f} KB)"
            )
    print("=" * 80)

    speedup = res_no_cache.total_time_ms / res_with_cache.total_time_ms
    print(
        f"\nSummary Metrics:"
        f"\n  • Without KV Cache : {res_no_cache.total_time_ms:7.1f} ms total | "
        f"Avg Decode ITL: {res_no_cache.avg_itl_ms:6.2f} ms/tok | "
        f"{res_no_cache.tokens_per_second:5.1f} tok/s"
        f"\n  • With KV Cache    : {res_with_cache.total_time_ms:7.1f} ms total | "
        f"Avg Decode ITL: {res_with_cache.avg_itl_ms:6.2f} ms/tok | "
        f"{res_with_cache.tokens_per_second:5.1f} tok/s"
        f"\n  • Overall Speedup  : {speedup:.2f}x faster"
    )


if __name__ == "__main__":
    main()
