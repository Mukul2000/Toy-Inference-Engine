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

---

## 6. Milestone 2 Design: Continuous Batching, and the Engine It Needs

This section tells Milestone 2 as a chain of problems. We start from the waste measured in Milestone 1.5, find the idea that removes it, and follow what that idea forces on the KV cache, on prefill, and finally on how the code is organized. Each choice comes with the alternative we turned down.

**The chain**: static batching wastes steps (6.1) → let requests come and go between steps (6.2) → give each request its own slot in the cache (6.3) → prefill new prompts into their slots, padded on the right (6.4) → one kind of work per step (6.5) → split *deciding* from *computing* (6.6) → keep what crosses between them small (6.7) → batching policies become one line (6.8). Then come the tensors (6.9), the code (6.10), the build plan (6.11), the tests (6.12) and what comes next (6.13). 6.14 recaps the decisions.

Background: [`notes/static_batching.md`](notes/static_batching.md), especially §4 on left vs. right padding.

### 6.1 Where We Are

Milestone 1.5 batches requests the simplest way. The prompts are left-padded into one `[B, T]` tensor, prefilled in one pass, then decoded in lockstep: at every step, every row writes its new token's K/V into the same cache column. One forward pass serves 8 rows for the price of one (47.9 ms vs. 48.2 ms per decode step), which bought a 5.5× throughput gain.

It also wastes work, in two ways we measured (§5.5):

- **Padding waste**: 26.8% of the prefill tokens were pads.
- **Tail waste**: 19.8% of the rows computed during decode belonged to requests that had already finished.

Tail waste is what this milestone attacks. Here is a small example that we follow through the whole section: 3 rows and 5 requests.

| Request | Prompt | New tokens needed |
| :--- | :--- | :--- |
| R1 | `A B C D E` | 6 |
| R2 | `F G H` | 2 |
| R3 | `I J K L` | 2 |
| R4 | `P Q` | 3 |
| R5 | `U V` | 3 |

Under static batching, R1–R3 form the first batch:

```
step:     0    1    2    3    4    5
row 0:    R1   R1   R1   R1   R1   R1✓
row 1:    R2   R2✓  ··   ··   ··   ··     ✓  = request finishes
row 2:    R3   R3✓  ··   ··   ··   ··     ·· = finished, but still computed
```

R2 and R3 are done after step 1. The batch is fixed until R1 finishes, so their rows are computed for four more steps, while R4 and R5 wait for a second batch: 3 more steps, 9 in total.

### 6.2 The Idea: Let Requests Come and Go Between Steps

A decode step is one forward pass that produces one token per row. Nothing in that math says the rows must belong to the same requests from one step to the next. Between two steps, we can drop a row whose request just finished and give its place to a waiting request:

```
step:     0    1    2    3    4    5    6
row 0:    R1   R1   --   R1   R1   R1   R1✓     -- = R1 waits one step (6.5 explains why)
row 1:    R2   R2✓  R4   R4   R4✓
row 2:    R3   R3✓  R5   R5   R5✓
```

R4 and R5 start at step 2 instead of step 6 and finish at step 4 instead of step 8. All five requests are done in 7 steps instead of 9. This is **continuous batching**. Orca (OSDI 2022) introduced it as *iteration-level scheduling*, and every modern engine does it.

The idea is simple, but today's code can't do it, for three reasons:

1. **One write column.** The cache keeps a single `current_seq_len` for the whole batch. From step 3 on, R1 and R4 decode side by side: R1 is deep into its sequence, R4 has barely started. They need different write columns in the same pass.
2. **Left padding needs a common start.** Left padding lines prompts up so they all end at the same column. That only works if everyone starts together; to line R4 up with R1, we would have to pad R4 by R1's entire history.
3. **The generator owns the loop.** `StaticBatchGenerator.generate()` cuts its requests into batches and runs each batch to the end. Nothing can join a batch in progress.

Sections 6.3–6.5 fix the first two. 6.6 fixes the third.

### 6.3 A Slot for Every Request

The cache tensor already has a batch axis: `[layers, B, max_seq_len, kv_heads, head_dim]`. In Milestone 1.5, row `b` of that axis belonged to request `b` for the whole run. We keep the tensor and change what that axis means: it becomes a **pool of slots**.

- A request **gets a free slot** when it is admitted, **owns it** while it runs, and **gives it back** when it finishes.
- Inside its slot, a request's tokens sit at columns `0, 1, 2, …` in order. So **column = position**, for every request, always.
- Each request has **its own length**: how many columns of its slot hold its tokens. There is no shared cursor.

The pool after step 2 of the example (lowercase letters are generated tokens: `a` is R1's first):

```
                       column:  0  1  2  3  4  5
slot 0, R1 (length 6):          A  B  C  D  E  a
slot 1, R4 (length 2):          P  Q  ░  ░  ░        ░ = left over from R2, the slot's previous owner
slot 2, R5 (length 2):          U  V  ░  ░  ░        ░ = left over from R3
```

This raises two questions:

- **Why not clear a slot when it is freed?** Nothing will read the leftovers. The new owner writes from column 0, and a request only ever attends to columns below its own length (6.4 shows the one mask rule that guarantees it). Clearing would be a write that buys nothing.
- **Why give every slot the same size?** Then a slot is one integer: the simplest allocation there is. The price is memory. One token's K and V take 2 × 30 layers × 3 heads × 64 dims × 4 bytes = 45 KiB, so a 512-column slot reserves 22.5 MiB even for a request that uses 40 columns. Paged KV (Milestone 3) fixes that by handing out small blocks instead of whole slots.

### 6.4 Getting a Prompt into Its Slot: Prefill

An admitted request has a prompt and an empty slot. Before it can generate, the model has to run over the whole prompt once. That pass writes K/V for every prompt token into the slot and produces the first new token. It is called **prefill**. Every later step feeds one token per request and is called a **decode**.

**One pass for all newcomers.** At step 0, three requests are admitted together. Prefilling them in one pass costs about as much as prefilling one, for the same reason batching helped decode: the weights stream through the CPU once for all rows. Measured on the 8 driver prompts: 228 ms in one pass, 554 ms one at a time.

**Padding, and which side.** One pass means one `[B, T]` rectangle. The prompts have different lengths, so the shorter ones need pads:

```
               left padding        right padding
column:        0  1  2  3  4       0  1  2  3  4
R1 (slot 0):   A  B  C  D  E       A  B  C  D  E
R2 (slot 1):   _  _  F  G  H       F  G  H  _  _
R3 (slot 2):   _  I  J  K  L       I  J  K  L  _
```

Both sides cost the same compute (228 vs. 231 ms for the 8 prompts). What differs is where the pads end up in each slot:

- **Left**: R2's two pads sit at the front of its slot for as long as R2 runs. Its tokens sit at columns 2–4 but are positions 0–2, so every later step needs R2's pad count to compute positions, plus a mask rule that skips the pads. That is Milestone 1.5's bookkeeping.
- **Right**: R2's tokens sit at columns 0–2, so column = position, as 6.3 requires. The pads sit at columns 3–4, which is where R2's next tokens go: its first decode writes column 3, on top of the first pad.

So we pad on the right. Two details make it work:

- **The length is the real prompt length.** After prefill, we record R2's length as 3, not the padded width 5. That is what makes its first decode write at column 3.
- **The next token comes from the last real token.** R2's prediction is read at column 2 (`length − 1`), not at the last column, which holds a pad.

**One mask rule.** A token at column `q` may attend to columns `k ≤ q` of its own slot. A request's real tokens always come before its pads and before any leftovers, so this one rule hides future tokens, pads, and a previous owner's data. Every pad also sits after at least one real token, so no query is left with nothing to attend to, and Milestone 1.5's NaN guard goes away. (Pad queries produce garbage; nothing reads it.)

The life of slot 1 shows all of this at once:

```
slot 1, column:   0  1  2  3  4
after step 0:     F  G  H  _  _     R2's prompt, right-padded to the batch width 5
after step 1:     F  G  H  f  _     R2's first token f lands on a pad. R2 is done: slot 1 is freed, bytes untouched
after step 2:     P  Q  H  f  _     R4's prompt overwrites columns 0-1. R4's length is 2: columns 2-4 are never read
after step 3:     P  Q  p  f  _     R4's first token p overwrites column 2
```

> [!NOTE] A third option: no pads at all
> Packing prompts end to end (`A B C D E F G H I J K L`) wastes nothing, but then attention must respect request boundaries inside one flat sequence: a custom kernel or a per-request loop. Chunked prefill (Milestone 4) is where that pays off.

### 6.5 Prefill and Decode Don't Share a Step

At step 2, R4 and R5 need prefill while R1 needs a decode. Could one pass do both? Not with a rectangle:

- **Pads.** A batch is one `[B, T]` rectangle, and every cell of it goes through all 30 layers. `T` is set by the longest row, so a 1-token decode row in a pass with a 56-token prompt gets 55 pads. One 56-token prompt plus 7 decode rows is 8 rows × 56 columns = 448 positions, of which 56 + 7 = 63 are real: 86% pads. Measured: that mixed pass takes 251 ms, while prefilling the prompt alone (84 ms) and then decoding all 8 rows (52 ms) takes 136 ms. Mixing makes even the running rows wait longer for their next token.
- **Overflow.** Each row writes K/V for all `T` of its columns, pads included: columns `start .. start + T − 1` of its slot. A pure decode step has `T = 1`, and a pure prefill step starts at 0 with prompts that fit (`add_request()` checks this), so neither can run past a slot's end. A mixed pass can: a request at length 500 in a 512-column slot, padded to a 56-token prompt's width, would write columns 500–555, and the 44 columns past 511 don't exist. This one is fixable (skip writing the pad columns), so it is a second reason, not the main one.

So every step is either all prefill or all decode, with prefill first:

1. If a slot is free and a request is waiting (and the batching policy allows it, 6.8), admit waiting requests in arrival order, one per free slot, and prefill them in one pass. Running requests sit this step out.
2. Otherwise, decode one token for every running request.
3. Either way, append each new token. A request that produced EOS or reached its `max_new_tokens` finishes, and its slot is free from the next step on.

Prefill also produces each newcomer's first token, so from the next step on, old and new requests decode together (step 3 in the example).

**The price is the pause**: R1 waits out step 2. Could running requests go first instead? If they always went first, a newcomer could only start once nobody was decoding, and that is static batching again. One paused step per admission is what continuous batching costs until chunked prefill (Milestone 4) puts prompt chunks and decode tokens into the same pass. vLLM's original scheduler (V0) made the same choice.

### 6.6 Splitting the Engine: Deciding vs. Computing

Look at everything one step now involves: keep a queue of waiting requests, hand out slots, track each request's length, choose prefill or decode, pad the rows, build positions and the mask, run the model, pick each row's next token, check for EOS and `max_new_tokens`, free slots.

Written as one generator class, every batching policy would need its own copy of that loop: the "new generator class per strategy" problem from Milestone 1.5. But the list splits cleanly in two:

| Deciding | Computing |
| :--- | :--- |
| who waits and who runs this step | padding the rows into a rectangle |
| which slot each request owns | positions and the attention mask |
| each request's length | the forward pass |
| when a request is done | picking each row's next token |
| | storing K/V in the slot pool |

The halves differ in every way that matters:

- **What they handle.** Deciding is bookkeeping over a few integers per request; it never needs to know that the model has 30 layers or 3 KV heads. Computing is all tensors and depends on the model's architecture.
- **What varies.** Naive, static and continuous batching differ only in deciding (6.8). Computing runs whatever rows it is handed, so one implementation serves all three.
- **How to test them.** Deciding can be tested against a fake model that returns made-up tokens: no torch, milliseconds. Computing can be tested by handing it rows directly.

So we split the code along this line, into two layers. Real engines draw the same line and call the halves the **control plane** and the **data plane**:

```
              LAYER 1 · CONTROL PLANE · deciding        engine/: pure Python, no torch
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ Engine      add_request() · step() · generate()                            │
  │ Scheduler   waiting queue · running list · free slots · admission policy   │
  │ Sequence    status · slot · length · output tokens                         │
  └────────────────────────────────────────────────────────────────────────────┘
          │ rows to run: (seq_id, slot, start, token_ids)      ▲ one token id per row
          ▼                                                    │
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ ModelExecutor      rows → input_ids + BatchMetadata → model → argmax       │
  │ KV slot pool       [layers, num_slots, max_seq_len, kv_heads, head_dim]    │
  │ TransformerModel   models/: the only model-specific code                   │
  └────────────────────────────────────────────────────────────────────────────┘
              LAYER 2 · DATA PLANE · computing          executor/ + models/: torch
```

> [!NOTE] The same split in vLLM and SGLang
> vLLM's engine core runs `schedule()` → `execute_model()` → `update_from_output()` every step, with the scheduler on one side and the model executor and its workers on the other. SGLang's scheduler builds a `ScheduleBatch`, and its model runner turns that into a `ForwardBatch` of tensors. When a new model architecture comes out, it is mostly the model definition that changes.

### 6.7 What Crosses the Line

A split is only as good as what has to cross it. Ours is crossed at two moments: at every step, and once at boot.

**Every step: rows down, token ids up.** Layer 1 sends a list of rows. Each row says *"run these tokens, starting at column `start` of slot `slot`"*:

```python
ScheduledSeq(seq_id=2, slot=1, start=0, token_ids=[F, G, H])   # R2's prefill: the prompt, from column 0
ScheduledSeq(seq_id=2, slot=1, start=3, token_ids=[f])         # R2's decode: one token, at its length
```

**`token_ids` = the tokens whose K/V is not in the slot yet.** They are this pass's input: the model embeds them, writes their K/V into the slot and attends over the slot. In a prefill row that is the whole prompt. In a decode row it is **the token sampled in the previous step**: the model produced it as an output, but it has never been an input, so its K/V doesn't exist yet. Feeding it in computes its K/V and, from it, the next token.

R2's whole life (prompt `F G H`, 2 new tokens, slot 1). At step 0 it shares the prefill pass with R1's 5-token prompt, so its row is padded to `[F G H _ _]`:

```
step  row sent down                  writes K/V to                  attends to     sampled from   returns
0     start 0, token_ids [F, G, H]   columns 0-4 (3-4 are its pads) columns 0..q   column 2 (H)   f
1     start 3, token_ids [f]         column 3                       columns 0-3    column 3 (f)   g   → R2 done
```

And slot 1 of the KV cache after each step. Each cell is one token's K and V vectors (in every layer, 45 KiB in total):

```
slot 1, column:    0    1    2    3    4    5   …  511
before step 0:     ·    ·    ·    ·    ·    ·   …   ·      · = empty (zeros, or a previous owner's data)
after step 0:      F    G    H    _    _    ·   …   ·      length 3: columns 3-4 hold pad K/V that nobody reads
after step 1:      F    G    H    f    _    ·   …   ·      f's K/V overwrites the first pad; length 4
                   └─── read by step 1 ───┘
```

Step 1 is where the cache pays off. Only `f` goes through the model. The K/V of `F G H` is read from the slot, not recomputed, and `f`'s query attends over columns 0–3.

Each output becomes the next input: `f` comes up at step 0 and goes back down at step 1. R2 stops after `g`, so `g` is never fed in and its K/V is never written; nobody needs it. Slot 1 goes back to the free list with `F G H f _` still in it, and the next owner overwrites it from column 0 (6.4).

**`start` = how many of the request's tokens are already in the slot** (its length, `num_computed_tokens` in Layer 1). It sets three things for the row's tokens, all the same number:

| | Rule | R2 at step 1 (`start = 3`) |
| :--- | :--- | :--- |
| Where K/V is written | columns `start .. start + T − 1` | column 3 |
| RoPE position | `start + 0, 1, …` | position 3 |
| How much history attention reads | columns `0 .. start + T − 1` (mask `k ≤ q`) | columns 0–3: `F G H f` |

A paused step doesn't change `start`. R1 (prompt `A B C D E`) sits out step 2, so its rows are `start 0 [A B C D E]`, then `start 5 [a]`, nothing at step 2, then `start 6 [b]` at step 3.

**Why `start` must be the real length.** If Layer 1 sent R2's padded width instead (`start 5` at step 1), `f` would be written at column 5, rotated as position 5, and attend over columns 3–4, which hold pads. R2's answer would change. Getting `start` right is what makes right padding work (6.4).

Layer 2 answers with one token id per row. Three choices hide in this:

- **One row shape for both kinds of step.** A prefill row starts at 0 and carries the prompt; a decode row starts at the request's length and carries one token. Layer 2 never asks which kind of step it is running. (vLLM's current scheduler has no prefill or decode phase either: each request just has some tokens computed and some still to compute.)
- **Token ids go up, not logits.** Logits are 49,152 floats per row; Layer 1 only needs the chosen token, to append it and check for EOS. So sampling (greedy argmax, for now) happens in Layer 2, as in vLLM.
- **Lengths come down every step.** Layer 2 keeps no per-request state between steps, only the K/V bytes. Each row's `start` tells it where that request's history ends.

**Why does Layer 1 deal with slots if Layer 2 holds the cache?** Because a cache involves two jobs:

- **Storing bytes** (Layer 2): model-shaped tensors that every attention layer reads. To Layer 2, a slot is just an index into the pool. It doesn't know which request holds slot 1, or whether slot 1 is free.
- **Bookkeeping** (Layer 1): the free list, and which request holds which slot.

The bookkeeping belongs in Layer 1 because the events that take and free slots happen there: admitting a request needs a free slot, and a slot frees when the stop check fires. If Layer 2 kept the free list, Layer 1 would have to ask "is there room?" before each admission and report "R2 is done" after each stop: the same state kept in two places.

```
Layer 1 (bookkeeping)               Layer 2 (bytes)
admit R4:   pop slot 1      ──►     "slot 1, start 0, [P, Q]"   writes columns 0-1
decode R4:  length is 2     ──►     "slot 1, start 2, [p]"      writes column 2
R4 stops:   push slot 1             not told: R4's K/V stays until the next
                                    owner overwrites it from column 0
```

An operating system splits memory the same way: the kernel decides which pages a process gets, and RAM only stores bytes. The [PagedAttention paper](https://arxiv.org/abs/2309.06180) draws vLLM like this too: one KV cache manager inside the central scheduler, and GPU workers that receive each request's block table with every step. In Milestone 3, our slot index becomes a block table.

**Once, at boot: limits up, pool shape down.** Layer 2 loads the model, so it knows the model's limits. Layer 1 decides how much room to reserve. The handshake is the first four lines of `Engine.__init__`:

```python
spec = executor.model_spec()                         # up:   max_model_len 8192, eos_token_id 2
assert max_seq_len <= spec.max_model_len             # a slot can't outgrow the RoPE table
executor.allocate_kv_cache(num_slots, max_seq_len)   # down: e.g. 8 slots × 512 columns × 45 KiB = 180 MiB
self.scheduler = Scheduler(num_slots, policy, spec.eos_token_id)
```

Different facts flow each way. The pool shape comes down from the `Engine` so that it is set in exactly one place, and the scheduler hands out exactly the slots the pool has. Configure the two layers separately and the scheduler could hand out slot 5 of a 4-slot pool.

> [!NOTE] vLLM's handshake carries more
> vLLM computes the pool size instead of taking it as an argument. Layer 2 reports each layer's KV layout and how much memory is left after a max-size dummy batch; Layer 1 divides that memory by the bytes per block and tells Layer 2 how many blocks to allocate. With several GPUs, one central count means every block id exists on every GPU. We can size the pool from a memory budget in Milestone 3.

### 6.8 Three Policies, One Line Apart

Here the split pays off. Naive, static and continuous batching run the same executor and the same step loop. They differ only in when waiting requests may be admitted:

```python
may_admit = self.policy is BatchingPolicy.CONTINUOUS or not self.running
# continuous: whenever a slot is free
# static:     only when the whole batch has finished
# naive:      static (or continuous) with num_slots = 1
```

The running example under each policy (`P` = prefill step, `D` = decode step, `✓` = finishes, `--` = paused, `·` = free slot):

```
static: 9 steps
step:     0    1    2    3    4    5    6    7    8
kind:     P    D    D    D    D    D    P    D    D
slot 0:   R1   R1   R1   R1   R1   R1✓  R4   R4   R4✓
slot 1:   R2   R2✓  ·    ·    ·    ·    R5   R5   R5✓
slot 2:   R3   R3✓  ·    ·    ·    ·    ·    ·    ·

continuous: 7 steps
step:     0    1    2    3    4    5    6
kind:     P    D    P    D    D    D    D
slot 0:   R1   R1   --   R1   R1   R1   R1✓
slot 1:   R2   R2✓  R4   R4   R4✓  ·    ·
slot 2:   R3   R3✓  R5   R5   R5✓  ·    ·

naive (1 slot): 16 steps, one request after another (6 + 2 + 2 + 3 + 3)
```

Unlike Milestone 1.5, the new static policy doesn't compute finished rows: their slots just sit empty until the batch drains. Steps are not equal in cost (a prefill step costs more than a decode step), so the counts show the shape of each policy, not its timing.

### 6.9 Two Steps, Traced Through the Tensors

What Layer 2 does with the rows, for R1 and R2 at steps 0 and 1. R3 rides along in slot 2; it is left out to keep the picture small.

`execute()` is four lines:

```python
input_ids = right_pad([row.token_ids for row in batch.seqs])   # [B, T]
meta = BatchMetadata.build(slot_ids, starts, lengths)           # positions, mask, last_idx
logits = self.model(input_ids, meta, self.kv_cache)             # [B, V]
return logits.argmax(dim=-1).tolist()                           # greedy sampling
```

**Step 0, prefill.**

```
rows from Layer 1 (plain ints)         BatchMetadata (Layer 2, tensors)
  R1: slot 0, start 0, [A B C D E]     input_ids = [[A, B, C, D, E],
  R2: slot 1, start 0, [F G H]                      [F, G, H, _, _]]   right-padded, T = 5
                                       slot_ids  = [0, 1]
                                       positions = [[0, 1, 2, 3, 4],
                                                    [0, 1, 2, 3, 4]]   start + 0 .. T-1
                                       last_idx  = [4, 2]              length - 1
                                       attn_mask = k <= q              S = 5 columns
```

R1 gets its first token `a` from column 4 and R2 gets `f` from column 2. Layer 1 records their lengths as 5 and **3**.

**Step 1, decode.**

```
rows                                   BatchMetadata
  R1: slot 0, start 5, [a]             input_ids = [[a], [f]]             T = 1
  R2: slot 1, start 3, [f]             slot_ids  = [0, 1]
                                       positions = [[5], [3]]
                                       last_idx  = [0, 0]
                                       S = max(start) + T = 6 columns
                                       R2's mask (its query is at column 3):
                                         key column:  0  1  2  3  4  5
                                                      ✓  ✓  ✓  ✓  ·  ·    4 is a pad, 5 is past R2's end
```

**Inside each attention layer**, the slot pool scatters the new K/V into each row's slot and gathers each row's history back:

```python
cols = starts[:, None] + arange(T)                # [B, T]: the columns each row writes
k_cache[layer, slot_ids[:, None], cols] = k_new   # scatter: each row into its own slot
return k_cache[layer, slot_ids, :S], ...          # gather: each row's history, [B, S, kv_heads, head_dim]
```

The gather is a copy (about 2 ms per step at 8 rows × 100 columns). Paged attention (Milestone 3) reads K/V in place instead. At the end of the model, `x[arange(B), last_idx]` picks each row's last real token for `lm_head`.

### 6.10 Putting It into Code

**Layer 1** (`engine/`) tracks each request as a `Sequence`:

```
add_request()
      │
      ▼
   WAITING
      │   admitted: gets a free slot, its prompt is prefilled
      ▼
   RUNNING  ↺  decode: one token per step
      │   EOS or max_new_tokens: its slot is freed
      ▼
   FINISHED
```

A sequence's length from 6.3 is `num_computed_tokens`: how many of its tokens have K/V in the slot. One rule then gives every row its input, prefill or decode:

```python
def next_input(self) -> list[int]:        # Sequence: the tokens not yet in the cache
    return (self.request.prompt_ids + self.output_ids)[self.num_computed_tokens:]
# just admitted: nothing computed yet   → the whole prompt  (prefill row, start 0)
# running:       all but the last token → the last token    (decode row, start = length)
```

A step is three calls, the same order as vLLM's engine core:

```python
def step(self) -> list[Sequence]:                    # Engine
    batch = self.scheduler.schedule()                # rows to run, by 6.5's prefill-first rule
    token_ids = self.executor.execute(batch)         # Layer 2: one token per row
    return self.scheduler.update(batch, token_ids)   # append, stop checks, free slots → finished
```

**The contract** (`contract.py`, no torch) holds the only types that cross between the layers:

```python
@dataclass(frozen=True)
class ModelSpec:            # boot, Layer 2 → Layer 1: limits Layer 1 must respect
    max_model_len: int      # 8192, the RoPE table
    eos_token_id: int       # 2, for the stop check

@dataclass(frozen=True)
class ScheduledSeq:         # one row of a step
    seq_id: int
    slot: int               # whose KV to read and extend
    start: int              # column (= position) of token_ids[0]
    token_ids: list[int]    # the tokens not yet in the cache

@dataclass(frozen=True)
class SchedulerOutput:      # every step, Layer 1 → Layer 2
    seqs: list[ScheduledSeq]

class Executor(ABC):        # everything Layer 1 knows about Layer 2
    def model_spec(self) -> ModelSpec: ...
    def allocate_kv_cache(self, num_slots: int, max_seq_len: int) -> None: ...
    def execute(self, batch: SchedulerOutput) -> list[int]: ...   # one token id per row
```

**The classes** on each side:

```python
# engine/: Layer 1
class Engine:
    def __init__(self, executor: Executor, policy=BatchingPolicy.CONTINUOUS,
                 num_slots=8, max_seq_len=512): ...                  # the boot handshake (6.7)
    def add_request(self, request: Request) -> int: ...              # rejects prompt + max_new_tokens > max_seq_len
    def step(self) -> list[Sequence]: ...
    def has_unfinished(self) -> bool: ...
    def generate(self, requests: list[Request]) -> list[RequestOutput]: ...   # add all, step until done

class Scheduler:                                                     # waiting queue, running list, free slots
    def add(self, seq: Sequence) -> None: ...
    def schedule(self) -> SchedulerOutput | None: ...
    def update(self, batch: SchedulerOutput, token_ids: list[int]) -> list[Sequence]: ...

# executor/: Layer 2
class ModelExecutor(Executor): ...     # loads TransformerModel, owns the slot pool

@dataclass
class BatchMetadata:                   # the tensor layout of one forward pass
    positions: Tensor                  # [B, T]        start + 0 .. T-1
    attn_mask: Tensor                  # [B, 1, T, S]  k <= q
    slot_ids: Tensor                   # [B]
    starts: Tensor                     # [B]
    last_idx: Tensor                   # [B]           length - 1

    @classmethod
    def build(cls, slot_ids, starts, lengths) -> "BatchMetadata": ...
```

Usage:

```python
engine = Engine(ModelExecutor(), policy=BatchingPolicy.CONTINUOUS, num_slots=3)
outputs = engine.generate([Request(prompt_ids, max_new_tokens=32) for prompt_ids in prompts])
```

**The model** changes in one way: `Attention` and `TransformerBlock` take `meta` instead of `positions, attn_mask`. `TransformerModel.forward(input_ids, meta=None, kv_cache=None)` returns logits at `meta.last_idx`. `meta=None` means fresh, unpadded rows, which keeps the experiments and the Hugging Face parity check working.

**Code layout:**

```
src/inference_engine/
├── __init__.py            driver: runs the three policies, checks the answer key, prints slot timelines
├── config.py              ModelConfig                                       (unchanged)
├── request.py             Request                                           (unchanged)
├── metrics.py             RequestOutput                                     (step metrics removed for now)
├── contract.py            ModelSpec, ScheduledSeq, SchedulerOutput, Executor        (new)
├── engine/                Layer 1: deciding, no torch                               (new)
│   ├── sequence.py        Sequence, SequenceStatus
│   ├── scheduler.py       Scheduler, BatchingPolicy
│   └── engine.py          Engine
├── executor/              Layer 2: computing                                        (new)
│   ├── model_executor.py  ModelExecutor
│   ├── batch_metadata.py  BatchMetadata
│   └── kv_cache.py        ContiguousKVCache, now the slot pool                      (moved)
└── models/                model definition; attention, layers and model take `meta`
tests/                                                                               (new)
├── data/reference_tokens.json   the answer key (Phase 0)
├── test_scheduler.py            Layer 1 against a fake executor: no model, no torch
├── test_executor.py             Layer 2 with a tiny random-weight model
└── test_engine.py               real model: every policy reproduces the answer key
```

**What goes away, and why.** All of it stays in git (commit `6000556`) and in §4–5.

- **The left-padded lockstep path**: `padding.py`, the pad mask and its NaN guard. Right padding into slots replaces it (6.4).
- **`generators/` and the `Generator` interface**: a strategy is now an admission policy (6.8), and `Engine.generate()` is the entry point.
- **Milestone 1's no-cache mode**: it showed why the cache matters, and that result stands (§4). It could come back as a "recompute every step" policy.
- **Step and waste metrics**: they come back with the benchmarks.

### 6.11 Building It in Four Phases

```
Phase 0 ──────────► Phase 1 ──────────► Phase 2 ──────────► Phase 3
answer key          Layer 2             Layer 1             cut over
```

0. **Answer key.** Run today's code on the 8 driver questions (32 new tokens each) and save the tokens to `tests/data/reference_tokens.json`. Every later phase rewrites code that produces tokens; the saved answers let each phase prove it changed nothing.
1. **Layer 2**: `contract.py`, `BatchMetadata`, the slot pool, `meta` through the model, `ModelExecutor`; delete `padding.py` and `generators/`. It goes first because it changes the model's math, the riskiest part, and it can be checked without Layer 1: drive the executor by hand (one prefill of all 8 prompts, then decodes) and compare against the answer key.
2. **Layer 1**: `Sequence`, `Scheduler`, `Engine`. Pure logic on top of a proven executor. Done when the fake-executor tests pass and all three policies reproduce the answer key.
3. **Cut over**: the driver runs all three policies, checks them against the answer key and prints each policy's slot timeline (the format of 6.8). Then write `notes/continuous_batching.md` and add the results to this section.

The driver is broken between Phases 1 and 3: Phase 1 changes the model's interface and deletes the generators the driver uses.

### 6.12 How We'll Know It Works

1. **Same answers.** Naive, static and continuous (8 requests on 3 slots, so slots are reused mid-run) produce the answer-key tokens, with the near-tie exception of §5.4.
2. **Layer 1 alone** (fake executor, no torch): static admits only into an empty batch; continuous refills a freed slot on the next step; a slot never holds two running requests; admission is first come, first served; `add_request()` rejects a request too big for a slot.
3. **Layer 2 alone** (`ModelConfig.tiny()`, random weights, runs in milliseconds): a row's output doesn't depend on which slot it is in or which rows share its batch (within float tolerance). Filling the pool with random values instead of zeros changes nothing, which proves leftovers are never read.
4. **The boundary holds**: nothing under `engine/` imports torch.
5. **HF parity** (Milestone 1) still holds through `meta=None`.

### 6.13 What's Left Out, and What Comes Next

Not in this milestone:

- **Benchmarks**: a working engine comes first; the 32-request workload comes later.
- **Arrivals over time**: every request is present at the start.
- **Preemption**: `add_request()` guarantees every request fits its slot, so nothing ever needs evicting.
- **Sampling beyond greedy**: greedy makes the answer-key check exact.
- Async or streaming APIs, prefix caching, packed prefill.

Every later feature is a new decision in Layer 1, new machinery in Layer 2, or both. The model definition stays the same:

| Feature | Layer 1 (deciding) | Layer 2 (computing) |
| :--- | :--- | :--- |
| Benchmarks | 32 requests, 8 slots, varied `max_new_tokens`; waste metrics | — |
| Paged KV (Milestone 3) | block manager: free list, block tables; pool sized from a memory budget | block pool; attention reads K/V through block tables |
| Chunked prefill (Milestone 4, §7) | token budget; split prompts; mix chunks with decodes | packed batches; attention in two groups |
| Prefix caching | hash and share blocks | nothing beyond paging |
| Temperature / top-p | per-request parameters in each row | a sampler |

### 6.14 Recap

| Decision | Why |
| :--- | :--- |
| Requests join and leave between steps | finished requests stop occupying rows; waiting ones start sooner (6.2) |
| One fixed-size slot per request; column = position | each request needs its own write column; a slot is one integer (6.3) |
| Newcomers prefilled together, right-padded | one pass is 2.4× faster than one at a time; decode overwrites right pads (6.4) |
| A step is all prefill or all decode | in a prefill rectangle, a decode row is mostly pads (6.5) |
| Two layers: deciding and computing | policies differ only in deciding, and deciding needs no model (6.6) |
| Rows down, token ids up | one row shape covers prefill and decode; ids are small, logits are big (6.7) |
| Slot bookkeeping in Layer 1, bytes in Layer 2 | Layer 1's decisions take and free slots (6.7) |
| Batching policies as one admission rule | they differ only in when requests may join (6.8) |

---

## 7. Milestone 4 Design: Chunked Prefill

This milestone is designed before paged KV (Milestone 3) because the two are independent: chunked prefill works on §6's contiguous slots, and paging changes nothing here. The story follows the same pattern as §6.

**The chain**: M2 makes every running request wait while a newcomer prefills (7.1) → prefill can resume from the cache, and a decode step has spare compute (7.2) → so each step gets a token budget: decodes first, then a chunk of a prompt (7.3), without changing any answer (7.4) → mixing needs a batch with no rectangle (7.5) → and attention on such a batch needs care (7.6). Then a traced step (7.7), the code (7.8), the decisions (7.9), the tests (7.10) and a recap (7.11).

### 7.1 The Problem M2 Leaves Behind

In M2, a step is all prefill or all decode (§6.5). While a newcomer prefills, every running request waits. With our 56-token prompts that's mild: one 84 ms pause where a decode step would take 52 ms.

The pause grows with the prompt. A 2,000-token prompt would take roughly 3 s on this CPU (a linear extrapolation, not measured), and for those 3 s, every user who is mid-answer sees their output freeze. Two latency metrics pull in opposite directions:

- **TTFT** (time to first token) of the newcomer wants its prefill done now, in one go.
- **ITL** (inter-token latency) of everyone else wants no step to be long.

### 7.2 Two Facts That Make Chunking Possible

**Prefill can resume from the cache.** Prefilling `[0:5]` and then `[5:12]` leaves the same K/V in the slot as prefilling `[0:12]` in one pass (7.4 shows why). The second piece is just a row that starts at column 5:

```python
ScheduledSeq(seq_id=4, slot=3, start=0, token_ids=prompt[0:5])    # chunk 1
ScheduledSeq(seq_id=4, slot=3, start=5, token_ids=prompt[5:12])   # chunk 2: the same row type
```

§6's contract already carries chunks. A piece of a prompt is called a **chunk**: the next tokens that aren't in the slot yet, capped in length. M2's prefill is the special case of one chunk that holds the whole prompt.

**A decode step has spare compute.** Decode streams all the weights for a handful of tokens (§5). Measured: `[8, 1]` takes 52 ms and `[1, 56]` takes 84 ms, so 7× the tokens cost 1.6× the time. Prompt tokens can ride along with the decodes almost for free.

### 7.3 The Idea: A Token Budget per Step

Each step gets a **token budget**: a cap on the total tokens in one pass, summed over all rows. The scheduler fills it in priority order:

1. One token for every request that is decoding, so nobody stalls.
2. The rest of the prompt that is already being prefilled, as much as fits.
3. Newcomers, admitted in arrival order while budget and free slots remain, each getting as much of its prompt as fits.

Only the last prompt admitted can be cut short, so at most one prefill is ever unfinished.

Example: budget 8. R1, R2 and R3 are decoding (lengths 7, 4, 5). R4 arrives with a 12-token prompt `p0 … p11` and gets slot 3:

```
step  decodes (1 token each)    budget left   R4's chunk                 tokens   R4's sample
k     R1 R2 R3                  5             start 0,  [p0 … p4]        8        discarded
k+1   R1 R2 R3                  5             start 5,  [p5 … p9]        8        discarded
k+2   R1 R2 R3                  5             start 10, [p10 p11]        5        q: R4's first token
k+3   R1 R2 R3 R4 [q]           4             —                          4
```

- R1–R3 get a token **every** step: no freezes.
- R4's first token arrives after 3 steps instead of 1: some TTFT traded for smooth ITL.
- Layer 2 still returns one token per row. For a chunk that ends mid-prompt, that token is the model's guess at the next **prompt** token, which we already know, so Layer 1 discards it.

Slot 3 fills chunk by chunk, with no pads:

```
column:       0  1  2  3  4  5  6  7  8  9  10  11  12
after k:      p0 p1 p2 p3 p4                                 num_computed_tokens 5
after k+1:    p0 p1 p2 p3 p4 p5 p6 p7 p8 p9                  num_computed_tokens 10
after k+2:    p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11          num_computed_tokens 12, prompt done
after k+3:    p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 q        decoding, like any other row
```

This is "stall-free scheduling" from the Sarathi-Serve paper (OSDI 2024). In vLLM, the budget is `max_num_batched_tokens`.

### 7.4 Why Chunking Doesn't Change the Answer

Follow `p7` at step k+1. In a one-shot prefill, its query attends to `p0 … p7`. Chunked, it attends to:

- `p0 … p4`: K/V read from the slot, written at step k;
- `p5 … p7`: K/V written in this step, before the read (the slot pool writes, then gathers, §6.9).

That's the same set of keys. And the K/V of `p0 … p4` are the same as in a one-shot pass, because under the causal mask a token's K/V in every layer depends only on the tokens up to it. This is the argument that makes the KV cache valid at all: a chunk is a prefill that resumes from the cache. (The float-level batch-invariance caveat of §5.4 still applies.)

### 7.5 Mixing Needs a Batch Without a Rectangle

§6.5 ruled that prefill and decode never share a step. That rule rested on three facts, and chunked prefill changes each one:

| Why mixing was bad in §6.5 | What changes here |
| :--- | :--- |
| **Pads.** The batch was a `[B, T]` rectangle, so a 1-token decode row was padded to the prompt's width: 448 positions for 63 real tokens. | **The batch is packed.** Real tokens are laid end to end with no pads, so a decode row costs one token. |
| **Overflow.** Pad columns could be written past the end of a slot. | Without pads, a row writes only its real tokens, so it can't run past its slot. |
| **Unbounded cost.** A whole prompt rode along, so a 2,000-token prompt meant a 2,000-token step. | **Bounded cost.** A chunk is at most the budget, so no step grows beyond it. |

So the two new ingredients have different jobs. The **packed layout** is what *allows* prefill and decode in one step; the **token budget** is what *bounds* each step.

**Packed** means the step's tokens form one flat sequence, and each token carries its own slot and position. Take step k+1 from 7.3 (R1–R3 decode `a'`, `b'`, `c'`):

```
token:      p5   p6   p7   p8   p9   a'   b'   c'
slot:       3    3    3    3    3    0    1    2
position:   5    6    7    8    9    8    5    6
```

Everything except attention works row by row and never asks which request a token came from: embedding, RMSNorm, the q/k/v/o projections, RoPE (each token rotated by its own position), the MLP, and the KV write (`k_cache[layer, slot[i], position[i]] = k[i]`). That's also where most of the compute is. `lm_head` runs only on each request's last token.

Attention is the only op that mixes tokens, so it's the only one that must respect request boundaries: each query may see only **its own slot**, up to **its own position**.

### 7.6 Attention on a Packed Batch

Three ways to do that were considered.

**B: one flat call with a block-diagonal mask.** Gather every request's history into one long key list and run one SDPA call, with a mask that lets each query see only its own request's keys, up to its position:

```
keys:        R1: cols 0-8      R2: 0-5     R3: 0-6      R4: cols 0-9
query p5     ·········         ······      ·······      ✓✓✓✓✓✓····
 …
query p9     ·········         ······      ·······      ✓✓✓✓✓✓✓✓✓✓
query a'     ✓✓✓✓✓✓✓✓✓         ······      ·······      ··········
query b'     ·········         ✓✓✓✓✓✓      ·······      ··········
query c'     ·········         ······      ✓✓✓✓✓✓✓      ··········
```

This is the shape real engines use (vLLM's one attention call per layer, with `query_start_loc` marking where each request's tokens start). But in plain PyTorch, SDPA with a dense mask computes **every** cell and then discards the masked ones, so the waste grows as tokens × total keys. Engines avoid that with *varlen* kernels (FlashAttention) that skip masked tiles: custom-kernel territory.

**C1: one padded rectangle, inside attention only.** Regroup the queries into `[B, T_max]`, run §6's rectangle attention, then pack the outputs again. The pads now exist only inside attention (the linears still see 8 tokens), but 12 of the 20 query rows here are pads.

**C2: two rectangles, grouped by shape.** Rows with more than one token form a **chunk group**, rows with one token form a **decode group**. The executor puts chunk tokens first, so each group is a contiguous slice of the packed tensor, and each group is an ordinary §6 rectangle:

```
packed:   p5 p6 p7 p8 p9 | a' b' c'
          └─ chunk group ─┘└decode group┘

chunk group   [1 × 5] queries, keys [1 × 10]   ← §6's prefill shape
decode group  [3 × 1] queries, keys [3 × 9]    ← §6's decode shape
```

Grouping is by shape, not by kind: a 1-token final chunk is mathematically a decode row (one query at position `start`), so it joins the decode group, and Layer 2 still never asks "prefill or decode?" (§6.7). vLLM V0 did the same with chunked prefill on: prefill tokens first, decode tokens after, a different kernel for each slice.

At a realistic size (8 slots: 7 requests decoding with ~500 tokens of history, plus a 64-token chunk starting at column 500; 71 tokens), by rough arithmetic:

| | Attention scores per head | Real | Useful |
| :--- | ---: | ---: | ---: |
| B, dense mask | 71 × 4,071 keys = 289k | 37.6k | 13% |
| C1, one rectangle | 8 × 64 queries × 564 keys = 289k | 37.6k | 13% |
| **C2, two groups** | 7 × 501 + 64 × 564 = 39.6k | 37.6k | **95%** |

B and C1 would spend about 20 GFLOP per step on attention, more than all the linear layers combined (~15 GFLOP for 71 tokens). C2 spends about 3 GFLOP.

**We choose C2.** It's the right shape for plain PyTorch SDPA, and it's the smallest step from §6: the contract, the slot pool and the rectangle attention all stay. B becomes attractive only with a varlen kernel that reads K/V straight from the cache, a natural partner for paged KV later.

### 7.7 Step k+1, Traced Through C2

The rows arrive in any order: `R1 [a'] · R2 [b'] · R3 [c'] · R4 start 5 [p5 … p9]`.

1. **Pack** (executor): multi-token rows first. `input_ids` is the flat `[p5 p6 p7 p8 p9 a' b' c']`, with per-token slots and positions as in 7.5.
2. **Outside attention**: embedding, norms, projections, RoPE and the MLP run on `[8, H]`.
3. **Inside attention**: write all 8 tokens' K/V to their slots, then run the two groups:

```
chunk group: queries [1 × 5], keys = slot 3, columns 0-9, mask k ≤ q
                 key col:  0  1  2  3  4  5  6  7  8  9
         p5 (pos 5)        ✓  ✓  ✓  ✓  ✓  ✓  ·  ·  ·  ·
         p6 (pos 6)        ✓  ✓  ✓  ✓  ✓  ✓  ✓  ·  ·  ·
         …
         p9 (pos 9)        ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓
                           └ from step k ┘└ this step ┘

decode group: queries [3 × 1], keys = slots 0, 1, 2, columns 0-8, mask k ≤ q   (§6's decode step)
```

4. **`lm_head`** on each row's last token: packed indices `[4, 5, 6, 7]`, i.e. `p9`, `a'`, `b'`, `c'`.
5. **Unsort**: return one token per row, in the order the rows arrived.

When several prompts are prefilled in one step (7.9, decision 1), the chunk group has several rows and pads them to the longest, inside attention only. At cold start, that is exactly §6's right-padded prefill.

### 7.8 Putting It into Code

**`Sequence`** gets one property, and `next_input()` gets a cap:

```python
@property
def is_prefilling(self) -> bool:                  # some prompt tokens are not in the slot yet
    return self.num_computed_tokens < len(self.request.prompt_ids)

def next_input(self, limit: int) -> list[int]:    # §6's rule, with a cap
    start = self.num_computed_tokens
    return (self.request.prompt_ids + self.output_ids)[start : start + limit]
```

**`Scheduler.schedule()`** fills the budget (7.3):

```python
def schedule(self) -> SchedulerOutput | None:
    budget, rows = self.token_budget, []
    def take(seq, limit):                         # add one row, spend its tokens
        nonlocal budget
        row = ScheduledSeq(seq.seq_id, seq.slot, seq.num_computed_tokens, seq.next_input(limit))
        rows.append(row); budget -= len(row.token_ids)

    for seq in self.running:                      # 1. every decoding sequence: 1 token
        if not seq.is_prefilling: take(seq, 1)
    for seq in self.running:                      # 2. finish the prefill already in progress
        if seq.is_prefilling and budget > 0: take(seq, budget)
    while budget > 0 and self.waiting and self.free_slots and self.may_admit():
        take(self.admit(self.waiting.popleft()), budget)   # 3. newcomers, with what is left
    return SchedulerOutput(rows) if rows else None
```

**`Scheduler.update()`** discards samples of partial chunks:

```python
for row, tok in zip(batch.seqs, token_ids):
    seq = self.seqs[row.seq_id]
    seq.num_computed_tokens += len(row.token_ids)
    if seq.is_prefilling:
        continue                                  # partial chunk: its sample is meaningless
    seq.output_ids.append(tok)                    # last chunk or decode: a real new token
    ...                                           # EOS / max_new_tokens checks, as in §6
```

What changes, relative to §6:

| Component | Change |
| :--- | :--- |
| `contract.py` | nothing: `ScheduledSeq` already carries chunks through `start` |
| `Sequence` | `is_prefilling`; `next_input(limit)` |
| `Scheduler` | the budget fill in `schedule()`; the partial-chunk discard in `update()` |
| `Engine` | `token_budget` argument; rejects `token_budget <= num_slots` |
| `ModelExecutor` | pack tokens, multi-token rows first; unsort the outputs |
| `BatchMetadata` | per-token `positions` and `slots` `[N]`, `last_idx` into the packed tokens, and up to two groups |
| `AttentionGroup` (new) | one rectangle: its slice of the packed tokens, slot ids, packed-to-rectangle index, `k ≤ q` mask |
| `Attention` | write all new K/V once, then one rectangle SDPA call per group |
| models outside attention | run on packed `[N, H]` instead of `[B, T, H]`; otherwise unchanged |
| KV slot pool | `update()` splits into `write()` (scatter by token, no pads) and `read()` (gather per group) |

With no budget, the same `schedule()` adds whole prompts to decode steps: continuous batching without §6.5's pause (packing removed the reason for it). Static batching with no budget runs exactly as in §6.

### 7.9 Decisions

| # | Decision | Why |
| :--- | :--- | :--- |
| D1 | **Several prompts per step**: leftover budget admits as many newcomers as fit | With one prompt per step, 8 empty slots fill in 8 steps; M2 fills them in 1. Several prompts cost only padding inside the chunk group's attention. |
| D2 | **Budget must exceed `num_slots`**; start around 64 for 8 slots, chosen by measurement | If decodes alone used up the budget, waiting prompts would never progress. vLLM has a similar rule. |
| D3 | **Layer 1 discards samples of partial chunks** | Keeps the contract unchanged. Costs one wasted `lm_head` row per partial chunk (576 × 49,152 multiply-adds, negligible). The alternative is a `sample` flag in `ScheduledSeq`. |
| D4 | **Chunking only for the continuous policy** | Static and naive keep §6's behaviour (an unlimited budget). Chunking exists to protect running requests from newcomers, and only continuous batching admits newcomers while others run. |

### 7.10 How We'll Know It Works

1. **Same answers**: with a small budget (so every prompt is chunked), continuous batching still reproduces §6's answer key.
2. **Layer 1 alone** (fake executor): no step exceeds the budget; every decoding request gets a token in every step; at most one prefill is unfinished at a time; partial-chunk samples are never appended.
3. **Layer 2 alone** (tiny model): prefilling a prompt in chunks of 1, 3 and 7 tokens gives the same logits as one pass, within float tolerance; a 1-token chunk lands in the decode group.
4. **Before choosing the budget**: measure step time against tokens per step, and measure the stall a long prompt causes under M2.

### 7.11 Recap

| Decision | Why |
| :--- | :--- |
| A token budget per step: decodes first, then prompt chunks | running requests never stall; prompts progress in bounded pieces (7.3) |
| A chunk is a row with `start > 0` | prefill resumes from the cache, so the contract doesn't change (7.2, 7.4) |
| Packed batches | no rectangle, so no pads: the reason §6.5 gave against mixing disappears (7.5) |
| Attention in two groups by shape (C2) | plain SDPA wastes almost nothing; reuses §6's rectangle attention (7.6) |
| Several prompts per step, budget > slots, discard partial samples, continuous only | 7.9 |
