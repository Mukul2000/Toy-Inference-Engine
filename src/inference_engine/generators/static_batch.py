import time

import torch
from einops import rearrange

from inference_engine.generators.base import Generator
from inference_engine.kv_cache import ContiguousKVCache
from inference_engine.metrics import GenerationResult, RequestOutput, StepMetric
from inference_engine.models.transformer import TransformerModel
from inference_engine.padding import left_pad
from inference_engine.request import Request


class StaticBatchGenerator(Generator):
    """
    Static batching: requests are taken in order, `max_batch_size` at a time. Each group is
    left-padded into one [B, T] tensor and runs from prefill until the LAST of them finishes.
    Nobody joins or leaves mid-batch, and the next group waits for the whole batch to finish.

    Controlled by `use_cache: bool`:
      - `use_cache=False`: Feeds every row's full sequence at every step (O(N^2) recomputation).
      - `use_cache=True`:  Feeds the padded prompts at Step 0 (Prefill), then ONLY each row's
                           newest token, [B, 1], at Steps 1..N (Decode).
    """

    def __init__(
        self, model: TransformerModel, max_batch_size: int = 8, use_cache: bool = True
    ):
        super().__init__(model)
        self.max_batch_size = max_batch_size
        self.use_cache = use_cache
        # Any id works for padding since pads are masked out of attention.
        self.pad_token_id = self.config.eos_token_id

    def generate(self, requests: list[Request]) -> GenerationResult:
        t_start = time.perf_counter()
        outputs: list[RequestOutput] = []
        step_metrics: list[StepMetric] = []
        for i in range(0, len(requests), self.max_batch_size):
            batch = requests[i : i + self.max_batch_size]
            outputs += self._run_batch(batch, t_start, step_metrics)
        return GenerationResult(
            outputs=outputs,
            total_time_ms=(time.perf_counter() - t_start) * 1000.0,
            step_metrics=step_metrics,
        )

    def _run_batch(
        self, batch: list[Request], t_start: float, step_metrics: list[StepMetric]
    ) -> list[RequestOutput]:
        """Runs one static batch from prefill until its last request finishes."""
        prompts = [r.prompt_ids for r in batch]
        input_ids, pad_lens = left_pad(prompts, self.pad_token_id)  # [B, T_max], [B]
        batch_size, prompt_len = input_ids.shape
        max_steps = max(r.max_new_tokens for r in batch)
        kv_cache = (
            ContiguousKVCache(self.config, batch_size, max_seq_len=prompt_len + max_steps)
            if self.use_cache
            else None
        )
        all_tokens = input_ids  # [B, T_max + steps so far], every row's full sequence
        finished = [False] * batch_size
        outputs = [RequestOutput() for _ in batch]

        for step_idx in range(max_steps):
            t0 = time.perf_counter()

            # Core difference between No-Cache and With-Cache:
            # - Without cache (or at Step 0 Prefill): pass every row's entire sequence
            # - With cache during Decode (step_idx > 0): pass ONLY each row's newest token
            feed_all = not self.use_cache or step_idx == 0
            step_tokens = all_tokens if feed_all else all_tokens[:, -1:]

            logits = self.model(step_tokens, pad_lens=pad_lens, kv_cache=kv_cache)
            next_ids = logits.argmax(dim=-1).tolist()  # greedy pick for every row
            now_ms = (time.perf_counter() - t_start) * 1000.0

            # Tail waste: a finished row keeps its slot and is still computed every step
            # until the whole batch is done. Its output is thrown away (replaced by a pad).
            num_active_rows = finished.count(False)
            for b, (request, out) in enumerate(zip(batch, outputs)):
                if finished[b]:
                    next_ids[b] = self.pad_token_id
                    continue
                out.token_ids.append(next_ids[b])
                if len(out.token_ids) == 1:
                    out.ttft_ms = now_ms
                out.latency_ms = now_ms
                finished[b] = (
                    next_ids[b] == self.config.eos_token_id
                    or len(out.token_ids) == request.max_new_tokens
                )

            step_metrics.append(
                StepMetric(
                    step_idx=len(step_metrics),
                    is_prefill=step_idx == 0,
                    batch_size=batch_size,
                    num_input_tokens=step_tokens.numel(),
                    num_pad_tokens=int(pad_lens.sum()) if feed_all else 0,
                    num_active_rows=num_active_rows,
                    step_latency_ms=(time.perf_counter() - t0) * 1000.0,
                    kv_cache_bytes=kv_cache.active_bytes if kv_cache is not None else 0,
                )
            )

            new_column = rearrange(torch.tensor(next_ids), "b -> b 1")
            all_tokens = torch.cat([all_tokens, new_column], dim=1)
            if all(finished):
                break

        return outputs
