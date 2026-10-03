# System Design Document: Modular LLM Inference Engine

**Author**: Mukul Singh  
**Status**: Draft / In Progress  
**Repository**: `/usr/local/google/home/mukulksingh/projects/inference-engine`

---

## 1. Background
Autoregressive Large Language Model (LLM) serving exhibits distinct computational characteristics:
- **Prefill (Prompt Processing)**: Compute-bound, parallel matrix multiplications (GEMM) over all prompt tokens.
- **Decode (Token Generation)**: Memory-bandwidth-bound matrix-vector operations (GEMV), generating one token at a time while repeatedly streaming weights and historical Key/Value (KV) activations from memory.

Traditional deep learning serving paradigms (static batching with 2D tensor padding) lead to extreme computational waste (up to 80% bubble overhead) and severe memory fragmentation. Modern production engines (e.g., vLLM, Orca, TensorRT-LLM, Google JetStream) address this through systems innovations: iteration-level continuous batching, paged virtual memory management, and chunked prefill co-scheduling.

This project implements a modular, clean-slate LLM inference engine from first principles in Python/PyTorch to demystify and evaluate these architectures.

---

## 2. Objectives
- **First-Principles Implementation**: Build a functional inference engine from the ground up, isolating each core systems innovation into modular, testable components.
- **Empirical Understanding**: Quantify and profile the physical bottlenecks of LLM inference (compute vs. memory bound, cache memory scaling, latency jitter, and fragmentation).
- **Separation of Concerns**: Architect a decoupled system where scheduling, memory management, and model execution operate independently.

---

## 3. Requirements

### 3.1 Functional Requirements (FR)
- **FR1 (Generation Modes)**: The engine must support:
  1. *Naive Recomputation*: Full prompt + past output forward pass at each step (no cache).
  2. *Contiguous KV Cache*: Incremental step-by-step forward pass maintaining a continuous KV tensor.
  3. *Continuous Batching*: Dynamic iteration-level scheduling of concurrent requests with variable prompt and output lengths.
  4. *Paged KV Memory*: Allocation and deallocation of KV cache in fixed-size physical blocks via a block table.
  5. *Chunked Prefill*: Budget-based prompt slicing co-scheduled with active decode steps.
- **FR2 (Model Compatibility)**: Support standard decoder-only transformer architectures (e.g., Qwen2 / Llama families) loaded via Hugging Face weights.
- **FR3 (Sampling & Output)**: Support greedy decoding ($T = 0$) for deterministic verification and temperature/top-$p$ sampling.
- **FR4 (Metrics & Telemetry)**: Provide instrumentation to record:
  - Time To First Token (TTFT).
  - Inter-Token Latency (ITL / TPOT) per step.
  - Overall system throughput (tokens/second).
  - Physical memory allocation across steps.

### 3.2 Non-Functional Requirements (NFR)
- **NFR1 (Portability)**: Core engine logic (scheduling, paging, chunking) must execute on commodity hardware (CPU / Apple Silicon / GPU) without mandatory dependencies on vendor-specific kernel languages (e.g., CUDA, Triton).
- **NFR2 (Correctness)**: Under greedy decoding, all generation paths (cached, paged, continuous) must produce token sequences strictly identical to Hugging Face reference outputs.
- **NFR3 (Modularity)**: Clean architectural separation between `Engine`, `Scheduler`, `BlockManager`, and `ModelBackend`.
- **NFR4 (Observability)**: Clear state representations (`SequenceStatus`, queue depths, block table mappings) suitable for step-by-step debugging.

---

## 4. Milestone 1 Design: Naive Baseline & KV Cache Mechanics

### 4.1 Problem Statement & Hypothesis
In autoregressive generation, computing token $t$ requires attention over all previous tokens $0 \dots t-1$.
- **Hypothesis 1 (No Cache)**: Recomputing the full sequence at every step incurs $O(N^2)$ cumulative compute per sequence. Latency per step increases linearly as context grows, leading to severe throughput degradation.
- **Hypothesis 2 (With KV Cache)**: Caching past key and value projections reduces step computation to $O(1)$ with respect to weights, keeping step latency relatively flat while memory consumption grows strictly linearly with sequence length:
  $$\Delta \text{Memory}_{\text{step}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{sizeof}(\text{dtype})$$

---

### 4.2 Architectural Components

```
                    ┌──────────────────────────────┐
                    │       ModelLoader / HF       │
                    │  (e.g., Qwen2.5-0.5B-CPU)    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                          NaiveGenerator                          │
│                                                                  │
│  ┌─────────────────────────────┐  ┌───────────────────────────┐  │
│  │   generate_without_cache    │  │  generate_with_cache      │  │
│  │   - Input: x[0:t]           │  │  - Input: x[t], past_kv   │  │
│  │   - Recomputes all tokens   │  │  - Appends to past_kv     │  │
│  └─────────────────────────────┘  └───────────────────────────┘  │
│                                                                  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Benchmark & Profiler    │
                    │  - Measures step latency     │
                    │  - Measures KV cache memory  │
                    │  - Verifies output parity    │
                    └──────────────────────────────┘
```

#### 1. `ModelLoader`
- Loads model weights and tokenizer from Hugging Face (default target: `Qwen/Qwen2.5-0.5B-Instruct` or similar compact model suitable for CPU execution).
- Configures execution device (`cpu` in our current test environment) and precision (`torch.float32` or `torch.bfloat16`).

#### 2. `NaiveGenerator`
Encapsulates single-sequence generation logic:
- **`generate_without_cache(prompt_ids, max_new_tokens)`**:
  - At step $k$ (where current sequence length is $L = L_{\text{prompt}} + k$):
    - Passes full tensor of shape `(1, L)` into `model(input_ids)`.
    - Extracts logits of the final token `logits[:, -1, :]`.
    - Samples next token $x_{L}$ and appends to the sequence tensor.
  - Measures elapsed time for each individual step $k$.

- **`generate_with_cache(prompt_ids, max_new_tokens)`**:
  - **Prefill Step**:
    - Passes prompt `input_ids` of shape `(1, L_{\text{prompt}})` into `model(input_ids, use_cache=True)`.
    - Receives initial logits and `past_key_values` tuple.
    - Samples first generated token $x_{0}$.
  - **Decode Steps**:
    - At step $k$: passes only the single latest token $x_{k-1}$ of shape `(1, 1)` and `past_key_values` into `model(...)`.
    - Receives new logits and updated `past_key_values`.
    - Samples $x_k$.
  - Measures elapsed time and size of `past_key_values` at each step $k$.

#### 3. `Benchmark & Profiler`
- Runs both methods on an identical test prompt with fixed length (e.g., 32 tokens prompt, 64 generated tokens).
- Compares outputs to assert exact token-for-token equality.
- Records:
  - Step index vs. Step Latency (ms).
  - Step index vs. KV Cache Memory (KB).
  - Total elapsed time and tokens per second.
- Formats results into a clean CLI table and summary plots/data.

---

### 4.3 Data Structures & Interfaces

```python
from dataclasses import dataclass
import torch

@dataclass
class StepMetric:
    step_idx: int
    token_id: int
    step_latency_ms: float
    kv_cache_bytes: int  # 0 for without_cache

@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    total_time_ms: float
    tokens_per_second: float
    step_metrics: list[StepMetric]
```

---

### 4.4 Verification & Acceptance Criteria
1. **Correctness Parity**:
   $$\text{tokens}_{\text{without\_cache}} == \text{tokens}_{\text{with\_cache}} == \text{tokens}_{\text{HF\_reference}}$$
2. **Computational Scaling**:
   - `without_cache` step latency must demonstrate a positive slope as step index increases (quadratic cumulative execution).
   - `with_cache` decode step latency must remain flat (constant time per token).
3. **Memory Accounting**:
   - Verify that KV cache memory growth per step matches the theoretical formula:
     $$\Delta \text{Bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 4 \text{ bytes (FP32)}$$

---

## 5. Milestone 1.5 Design: Static Batching

Background and derivations: [`notes/static_batching.md`](notes/static_batching.md).

### 5.1 Problem Statement & Hypothesis
- At $B = 1$, every decode step streams all ~540 MB of weights from RAM to do one multiply-add per weight. Running requests in parallel threads does not help: each forward pass pays that cost again.
- **Hypothesis**: stacking $B$ requests as rows of one tensor turns $B$ matrix-vector products into one matrix-matrix product. Step latency stays roughly flat while $B$ is small, so throughput scales almost linearly with $B$.
- **Expected costs**: *padding waste* (prompts padded to the longest) and *tail waste* (finished rows hold their slot until the longest row finishes).

### 5.2 Design
- **Layout**: left-padded `[B, T]`. Every row ends at the same column, so decode moves in lockstep with one shared write column.
- **Only attention needs batch awareness.** Embeddings, RMSNorm, projections, MLP and `lm_head` work row-wise and need no change.

| Component | Change |
| :--- | :--- |
| `padding.py` (new) | `left_pad()` (engine side) and `build_positions_and_mask()` (model side) |
| `ContiguousKVCache` | buffer `[L, B, S, n_kv, d]`, written by **column** at a single `current_seq_len` |
| `RotaryEmbedding` | per-request positions `[B, T]` |
| `Attention` | `[B, T, heads, d]` shapes, explicit boolean `attn_mask` instead of `is_causal` |
| `TransformerModel` | `forward(input_ids [B,T], pad_lens [B], kv_cache) -> logits [B, V]`, `lm_head` on the last column only |
| `generators/` (new package) | `Generator` interface (see 5.3). `StaticBatchGenerator` keeps `use_cache`. `NaiveGenerator` is `StaticBatchGenerator` with `max_batch_size = 1` |

**Invariants**:
1. **Storage column ≠ position**: `position = column − pad_lens[b]`.
2. **Mask**: a query may attend a key iff (causal **and** key is not a pad) **or** key is the query itself. The last clause is the NaN guard for pad queries.
3. **Lockstep**: one `current_seq_len` for the whole batch. Finished rows keep computing until all rows finish.

### 5.3 Interfaces
Every generation strategy implements one interface, so the driver and benchmarks do not change when a strategy is added (Milestone 2 adds `ContinuousBatchGenerator`):

```python
class Generator(ABC):                                     # generators/base.py
    @abstractmethod
    def generate(self, requests: list[Request]) -> GenerationResult: ...

Request(prompt_ids: list[int], max_new_tokens: int)       # request.py
RequestOutput(token_ids, ttft_ms, latency_ms)             # metrics.py, one per request
GenerationResult(outputs, total_time_ms, step_metrics)    # metrics.py, one per run

StaticBatchGenerator(model, max_batch_size=8, use_cache=True)
NaiveGenerator(model, use_cache=True)                     # StaticBatchGenerator, max_batch_size=1
```

- **Request-level, not batch-level**: callers hand over requests, never batches. How requests are grouped into forward passes is the scheduling policy, the very thing that differs between strategies. Continuous batching has no fixed batch at all.
- **Per-request `max_new_tokens`**. TTFT and latency are measured from the start of `generate()`, so time spent waiting behind other requests counts.
- **`abc.ABC`, not `typing.Protocol`**: we own every implementation and want an explicit "is-a" relationship with a shared `__init__`. A subclass that forgets `generate()` fails at construction.

Model side:
```python
input_ids, pad_lens = left_pad(prompts, pad_token_id)                # [B, T], [B]
positions, attn_mask = build_positions_and_mask(pad_lens, start, T)  # [B, T], [B, 1, T, start + T]
logits = model(input_ids, pad_lens=pad_lens, kv_cache=kv_cache)      # [B, V]
```

### 5.4 Verification & Acceptance Criteria
1. **Batched vs. alone parity** (refines NFR2 for batched paths): each request's tokens in the batch equal its tokens when run alone. Bit-identical logits are **not** required: matmul results depend on batch shape (batch invariance), so any divergence must occur at a near-tie (small top-2 logit gap).
2. **Cache vs. no-cache parity** in batched mode.
3. **Negative control**: without the pad mask, padded rows must diverge.
4. **Throughput**: report speedup over serving the same requests one at a time, plus padding and tail waste.

### 5.5 Results
8 chat requests (prompts 35–56 tokens, up to 32 new tokens), SmolLM2-135M, FP32, CPU:

| | One at a time ($B = 1$) | Static batch ($B = 8$) |
| :--- | ---: | ---: |
| Total time | 10,197 ms | 1,845 ms |
| Throughput | 20.3 tok/s | **112.2 tok/s (5.53×)** |
| Decode step latency | 48.2 ms (1 row) | 47.9 ms (8 rows) |
| Avg TTFT | 4,022 ms | 354 ms |
| Avg request latency | 5,223 ms | 1,549 ms |

- **Latency**: all 8 requests are waiting at $t = 0$ and times are measured from there, so the $B = 1$ numbers include queueing behind earlier requests. With every request present up front, static batching wins on every metric. Its weakness (a request arriving mid-batch waits for the whole batch) needs arrivals over time to show up.

- **Parity**: 8/8 requests token-identical to running alone. Cache matches no-cache, batched and alone. Logits match a Hugging Face left-padded batch within 4.2e-5, with the same argmax.
- **Negative control**: removing the pad mask shifts padded rows' logits by 0.2–0.9.
- **Waste**: 26.8% of prefill tokens were pads (5.27 MiB of KV cache). 19.8% of decode row-slots belonged to already-finished requests. These are the baselines for Milestone 2 (continuous batching).
