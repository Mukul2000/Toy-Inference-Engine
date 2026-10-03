### 1. What Is It?

**Static batching** runs $B$ independent requests through **one forward pass** by stacking them as rows of a single tensor: `input_ids` goes from `[T]` to `[B, T]`. The batch is formed once, runs prefill, then decodes in lockstep until **every** row has finished. No request can join or leave while the batch is running ("static").

It is the simplest way to serve more than one request at a time, and the baseline that continuous batching (Milestone 2) improves on.

---

### 2. Why Batch At All? (Measured)

Running the model is dominated by streaming the **weights** (~540 MB for SmolLM2-135M, far larger than the 48 MB L3 cache) through the CPU, not by arithmetic. At $B = 1$ every weight is loaded from RAM, used for a single multiply-add, and evicted.

Three ways to serve $B$ requests that each need one token, measured on the 24-vCPU box (median of 7 runs):

| Strategy | B = 8 | B = 16 |
| :--- | ---: | ---: |
| (a) $B$ forward passes, one after another | 388 ms | 794 ms |
| (b) $B$ forward passes in $B$ Python threads | 351 ms | 915 ms |
| (c) **1 forward pass with $B$ rows** | **49 ms** | **61 ms** |

A single-row pass costs ~48 ms, so 8 batched requests cost the same as 1.

```
(a)/(b):  W·x₁, W·x₂, …, W·x₈   → W streamed from RAM 8 times (matrix-vector, GEMV)
(c):      W·[x₁ x₂ … x₈]        → W streamed once, reused 8 times (matrix-matrix, GEMM)
```

> [!info] Why threads/processes do not help
> One pass already uses every physical core (PyTorch splits each matmul across them). The cost that batching removes is the **per-pass fixed cost** (weights streamed, hundreds of ops launched), and $B$ separate passes pay it $B$ times no matter how they are scheduled. The weight reuse in (c) happens *inside* the matmul, so it can only be arranged by making the requests rows of the same tensor. Running model copies in parallel is how you scale across machines; batching is how you get more out of one.

---

### 3. Which Layers Have To Know About the Batch?

For each op ask: *does row $i$'s output depend on any other row?*

| Op | Mixes rows? | Change needed |
| :--- | :--- | :--- |
| `embed_tokens`, RMSNorm | no | none |
| q/k/v/o projections, MLP, `lm_head` | no | none (`nn.Linear` already accepts `[B, T, H]`) |
| RoPE | no, but needs each request's **own** positions | `positions`: `[T]` → `[B, T]` |
| **Attention** $\text{softmax}(QK^\top)V$ | **yes, the only one** | batch dim + mask: never across requests, never onto padding |
| [[KV Cache]] | stores each request's history | add a batch dim |

What breaks without this: feed two requests' decode tokens as one 1D sequence `[x_A, y_B]`. The model sees one sequence, the causal mask lets `y_B` attend to `x_A`, and request B's answer now depends on request A's text.

> [!tip] The irony
> The `nn.Linear` layers that need **no** change are where all the speedup comes from. Attention, the op that **must** change, gains almost nothing. Batching helps only when requests share an input, and the shared input is the weights. Attention has no weights: its $Q, K, V$ belong to one request. So as contexts grow, attention takes a bigger share of each step and batching helps less. This is the root of why the KV cache becomes *the* bottleneck at scale, and why paged attention exists.

---

### 4. Padding: Why and Where

A tensor is a rectangle; prompts are not. Shorter prompts must be padded to the longest prompt's length $T_{max}$. The side you pad on looks cosmetic, but it decides **what state the engine must track per row.**

Worked example used throughout. Row 0: `A B C D` (4 tokens). Row 1: `E F` (2 tokens). `_` is a pad.

> [!note] Notation
> $B$ is the batch size; $b \in \{0, \dots, B-1\}$ is the **batch index** (which row / which request), the same way $T$ is the sequence length and $t$ one position in it. Per-row quantities are length-$B$ vectors indexed by $b$: here `pad_len = [0, 2]` and `len = [4, 2]`, so `pad_len[1] = 2` is "row 1 has two pads".

```
            LEFT padding                 RIGHT padding
column:     0  1  2  3                   0  1  2  3
row 0:      A  B  C  D                   A  B  C  D
row 1:      _  _  E  F                   E  F  _  _
```

#### 4.1 Prefill: both work, bookkeeping differs

**Positions (what RoPE sees).** `E` must be rotated as position 0 and `F` as 1, exactly as if the prompt ran alone, or the output changes.

- Right: `position = column`. Free.
- Left: `position = column − pad_len[b]`, so row 1 is `[_, _, 0, 1]`. One subtraction, but **storage column ≠ position**, which breaks the Milestone 1 assumption in `ContiguousKVCache` (it uses the position as the storage index).

**Mask.** Causal plus "never attend to a pad as a key" (✓ = may attend):

```
LEFT, row 1 (q = row, k = col)      RIGHT, row 1
       k0 k1 k2 k3                        k0 k1 k2 k3
q0 _   ✓  ·  ·  ·                   q0 E  ✓  ·  ·  ·
q1 _   ·  ✓  ·  ·                   q1 F  ✓  ✓  ·  ·
q2 E   ·  ·  ✓  ·                   q2 _  ·  ·  ✓  ·
q3 F   ·  ·  ✓  ✓                   q3 _  ·  ·  ·  ✓
```

Same shape of problem on both sides. Pad queries see only themselves (see the NaN guard in section 5); their outputs are garbage that nobody reads.

**Reading the logits.** We want the hidden state of the **last real token** per row.

- Left: always column $T_{max}-1$, so `x[:, -1]` works for every row.
- Right: column `len[b] − 1`, different per row. Needs a gather: `x[arange(B), lens − 1]`. Use `x[:, -1]` by mistake and row 1 predicts from a *pad's* hidden state.

> [!warning] This is the bug behind Hugging Face's warning
> *"A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'`."* Encoder models (BERT) read every position, so right padding is harmless there; decoders read the last position, so the last column must be real.

#### 4.2 Decode: where the two strategies split

Each row produces a token and its $K, V$ must be written into the cache. **Where?**

```
LEFT                                    RIGHT
column:  0  1  2  3  4  5               column:  0  1  2  3  4  5
row 0:   A  B  C  D  X₁ X₂              row 0:   A  B  C  D  X₁ X₂
row 1:   _  _  E  F  Y₁ Y₂              row 1:   E  F  Y₁ Y₂
                     ↑  ↑                              ↑  ↑
         one write column per step      row 0 writes col 4, row 1 writes col 2
```

**Left padding = lockstep.** All rows end at the same column, so each step:
- writes $K, V$ at **one shared column**: `cache[layer, :, col] = k_new`
- positions are `col − pad_len` (a fixed vector computed once at prefill)
- mask for key column $c$ is simply `c ≥ pad_len[b]`
- `current_seq_len` stays a **single integer**, as in Milestone 1

**Right padding = per-row cursors.** Each row sits at a different length, so each step:
- writes with a scatter: `cache[layer, b, len[b]] = k_new[b]` per row
- positions are `len[b]`, changing every step
- mask for key column $c$ is `c < len[b]`
- the attention slice must cover `max(len)`, so short rows carry trailing junk columns

> [!info] Right padding is already ragged batching
> Nothing above is impossible; every piece of state just goes from one scalar to a length-$B$ vector: a per-sequence length, a per-sequence write slot, a per-sequence position. That vector **is** the core data structure of continuous batching. Left padding is the trick that lets a static batch skip it and move in lockstep. Right padding + per-row cursors is one step away from Milestone 2.

#### 4.3 What each strategy wastes

| | Left padding | Right padding |
| :--- | :--- | :--- |
| Prefill compute on pads | yes, `pad_len[b]` tokens per row | same |
| KV memory on pads | **yes, forever**: pad columns sit in the cache for the life of the request | no: decode tokens overwrite the pad slots, each row stays contiguous |
| Decode attention over pads | yes, masked but still computed by dense SDPA | no |
| Per-step state | 1 scalar + a fixed `pad_len` vector | `len` vector updated every step |
| Logits read | `x[:, -1]` | gather by `len − 1` |

#### 4.4 The third option: no padding (packed / ragged)

Flatten all real tokens into one 1D tensor `[A B C D E F]` plus boundary metadata (`cu_seqlens = [0, 4, 6]`) and make attention respect the boundaries. Zero waste, but attention is no longer one SDPA call over a rectangle: it needs a custom kernel (vLLM, FlashAttention `varlen`) or a per-request loop / block-diagonal mask in plain PyTorch. This is the natural shape for Milestone 2, where the cache is paged anyway.

---

### 5. Does Padding Change the Answer?

With correct positions and a correct mask: **no** (up to float-ordering effects, section 7). Three things keep that true:

1. **Pads are never visible as keys to real tokens.** The mask removes them from the softmax entirely, so the pad's $K, V$ values do not matter. They are written into the cache but never read with nonzero weight. (This also answers "which token id do I pad with?": any id, since its embedding is masked out. Conventionally `eos` or 0.)
2. **Real tokens get their standalone positions** via RoPE, so `E` is rotated as position 0 either way.
3. **Pads as queries must see at least themselves.** A fully-masked softmax row has no finite entries and SDPA returns NaN. That NaN would land in the pad's attention output, flow through the residual stream into the pad's next-layer $K, V$, and leak into real rows because $0 \times \text{NaN} = \text{NaN}$ even under a mask. OR-ing the mask with the diagonal prevents it (Hugging Face calls this "unmask unattended").

> [!warning] `is_causal=True` is not enough any more
> `F.scaled_dot_product_attention(is_causal=True)` builds a top-left-aligned triangle and knows nothing about pads. Batched attention always passes an **explicit boolean mask** of shape `[B, 1, T_q, T_kv]`.

---

### 6. Shapes (Left-Padded Static Batch)

$B$ requests, longest prompt $T$, cache capacity $S$, SmolLM2 sizes ($H=576$, 9 Q heads, 3 KV heads, $d=64$, 30 layers, $V = 49152$):

| Tensor | Prefill | Decode step |
| :--- | :--- | :--- |
| `input_ids` | `[B, T]` | `[B, 1]` |
| `positions` | `[B, T]` (`col − pad_len`) | `[B, 1]` |
| `x` | `[B, T, 576]` | `[B, 1, 576]` |
| `q` | `[B, T, 9, 64]` | `[B, 1, 9, 64]` |
| `k_new`, `v_new` | `[B, T, 3, 64]` | `[B, 1, 3, 64]` |
| KV cache (per layer) | `[B, S, 3, 64]` | same buffer, one more column filled |
| attention mask | `[B, 1, T, T]` | `[B, 1, 1, t+1]` |
| `lm_head` input | `x[:, -1]` → `[B, 576]` | `[B, 576]` |
| logits | `[B, 49152]` | `[B, 49152]` |

Running `lm_head` only on the last column saves ~20% of prefill FLOPs: `lm_head` is $576 \times 49152 \approx 28$M parameters, vs. ~106M for all 30 layers.

---

### 7. Batched Results Are Not Bit-Identical

A row computed inside a batch does **not** bit-match the same row computed alone (checked for $B = 2 \dots 64$):

| Module | max abs diff (batched vs. alone) |
| :--- | ---: |
| `q_proj`, `lm_head` | ~1e-5 |
| layer-0 MLP | ~5e-4 |

The BLAS library picks a different blocking strategy per matrix shape, so the partial sums are added in a different order, and float addition is not associative. Greedy tokens almost always still match, but can flip at a near-tie, after which the rest of the output diverges.

Consequences:
- "Strictly identical" parity (NFR2 in the design doc) cannot hold bit-for-bit for batched runs. Milestone 1's cache vs. no-cache check already lived with this (different shapes, same tokens).
- A fair correctness check is: *tokens match the request run alone; any divergence must occur at a near-tie (small top-2 logit gap).*
- This is the **batch invariance** problem. Production engines expose opt-in batch-invariant kernels (vLLM `VLLM_BATCH_INVARIANT=1`) at a 20–60% throughput cost. See *Defeating Nondeterminism in LLM Inference* (Thinking Machines, 2025).

---

### 8. The Two Wastes Static Batching Leaves Behind

These are what [[Continuous Batching]] (Milestone 2) exists to remove, and what Milestone 2 should measure against:

1. **Padding waste.** Compute (prefill) and KV memory (left padding) spent on `_` tokens. Grows with the spread of prompt lengths in the batch.
2. **Tail waste.** A row that hits EOS early keeps its slot and computes junk until the **longest** row finishes. No new request can take the slot. Grows with the spread of output lengths.

```
step:    0  1  2  3  4  5  6  7
row 0:   X  X  X  X  X  X  X  X   ◄─ longest output dictates batch lifetime
row 1:   Y  Y  Y  EOS ·  ·  ·  ·  ◄─ 4 wasted decode slots
row 2:   Z  EOS ·  ·  ·  ·  ·  ·  ◄─ 6 wasted decode slots
```

> [!tip] Verified in code
> `uv run inference-engine`: 8 chat requests (prompts 35–56 tokens, up to 32 new tokens). Serving them one at a time took 10.2 s (20.3 tok/s); one static batch took 1.8 s (112.2 tok/s), a **5.53× speedup**. All 8 outputs were token-identical to running alone. A decode step with 8 rows cost 47.9 ms, about the same as with 1 row (48.2 ms). Because all 8 requests were waiting from the start, latency improved too: average TTFT fell from 4.0 s (each request queued behind the earlier ones) to 0.35 s. Measured waste: **26.8%** of prefill tokens were pads, and **19.8%** of decode row-slots belonged to requests that had already finished.

> **One-Sentence Rule**: _Static batching turns $B$ weight-streaming GEMVs into one GEMM, at the price of padding every prompt to the longest and holding every slot until the longest output ends._
