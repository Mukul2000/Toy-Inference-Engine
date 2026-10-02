import time
import torch

from inference_engine.kv_cache import ContiguousKVCache
from inference_engine.metrics import GenerationResult, StepMetric
from inference_engine.models.transformer import TransformerModel


class NaiveGenerator:
    """
    Milestone 1 Autoregressive Generator controlled by `use_cache: bool`:
      - `use_cache=False`: Feeds `all_tokens` at every step (O(N^2) recomputation).
      - `use_cache=True`:  Feeds `all_tokens` at Step 0 (Prefill), then ONLY
                           `[all_tokens[-1]]` (1 token) at Steps 1..N (Decode).
    """

    def __init__(self, model: TransformerModel):
        self.model = model
        self.config = model.config

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        use_cache: bool = True,
    ) -> GenerationResult:
        kv_cache = (
            ContiguousKVCache(self.config, max_seq_len=len(prompt_ids) + max_new_tokens)
            if use_cache
            else None
        )
        all_tokens = list(prompt_ids)
        generated_ids: list[int] = []
        step_metrics: list[StepMetric] = []

        t_start = time.perf_counter()

        for step_idx in range(max_new_tokens):
            t0 = time.perf_counter()

            # Core difference between No-Cache and With-Cache:
            # - Without cache (or at Step 0 Prefill): pass the entire sequence `all_tokens`
            # - With cache during Decode (step_idx > 0): pass ONLY the 1 newest token!
            step_tokens = (
                all_tokens if (not use_cache or step_idx == 0) else [all_tokens[-1]]
            )
            input_ids = torch.tensor(step_tokens, dtype=torch.long)

            logits = self.model(input_ids, kv_cache=kv_cache)
            next_token_id = int(logits[-1].argmax().item())

            step_ms = (time.perf_counter() - t0) * 1000.0
            step_metrics.append(
                StepMetric(
                    step_idx=step_idx,
                    num_input_tokens=len(step_tokens),
                    token_id=next_token_id,
                    step_latency_ms=step_ms,
                    kv_cache_bytes=kv_cache.active_bytes if kv_cache is not None else 0,
                )
            )

            all_tokens.append(next_token_id)
            generated_ids.append(next_token_id)

            if next_token_id == self.config.eos_token_id:
                break

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return GenerationResult(
            token_ids=generated_ids,
            total_time_ms=total_ms,
            tokens_per_second=(len(generated_ids) / (total_ms / 1000.0)),
            step_metrics=step_metrics,
        )
