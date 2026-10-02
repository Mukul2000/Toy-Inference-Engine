### 1. What Is It?

An in-memory buffer for [[Self-Attention]] (one per Transformer layer, holding every KV head) that stores the **Key ($k_j$)** and **Value ($v_j$)** vectors of all previously processed tokens ($j = 0, 1, \dots, t-1$) during autoregressive generation.
### 2. Why Do We Need It & How Does It Help?

- **Without KV Cache**: To generate token $t+1$, you must feed the **entire sequence** $[x_0, \dots, x_t]$ back into the model and recompute $Q, K, V,$ and MLP layers for every old token from scratch. Because of the causal mask, old tokens cannot see new tokens—so you are recomputing the exact same numbers over and over ($O(N^2)$ linear/MLP compute).
- **With KV Cache**: At step $t$, you feed **only the 1 newest token $[x_t]$** into the model, compute its $(q_t, k_t, v_t)$, append $(k_t, v_t)$ to the cache, and attend $q_t$ over the cached history ($O(1)$ linear/MLP compute per step).

> **Prerequisite: the causal mask** (see Stage 5b in [[Self-Attention]]). Token $j$ may only attend to tokens $0 \dots j$, so its hidden state in **every layer**, and therefore its $k_j$ and $v_j$ in every layer, depends only on tokens $0 \dots j$. Adding new tokens later can never change them, which is why they are safe to cache (diagram in section 6). Without the mask, every new token would change the attention output of all old tokens, which changes their hidden states going into the next layer, so all old $K$, $V$ from layer 1 upward would go stale and the cache would be useless.

---

### 3. Why Is It a "KV" Cache (Not a "QKV" or "V" Cache)?

Look at the Attention matrix at **Step 3** (when token `t3` arrives to predict `t4`). This grid is **one slice** of the Stage 5 score tensor in [[Self-Attention]]: one sentence, one head, and 4 tokens instead of 32 (with the Stage 5b causal mask already applied):

```
                  Keys (k0, k1, k2, k3)
                  k0      k1      k2      k3
Queries:
  q0 (token 0) [ q0·k0   -inf    -inf    -inf  ]  ◄── Already used at Step 0 (never changes)
  q1 (token 1) [ q1·k0   q1·k1   -inf    -inf  ]  ◄── Already used at Step 1 (never changes)
  q2 (token 2) [ q2·k0   q2·k1   q2·k2   -inf  ]  ◄── Already used at Step 2 (never changes)
  ──────────────────────────────────────────────
  q3 (token 3) [ q3·k0   q3·k1   q3·k2   q3·k3 ]  ◄── THE ONLY ROW NEEDED AT STEP 3!

```

- In **prefill**, all 4 rows are computed at once (the full square, as in the self-attention stages).
- In **decode**, only the **last row** is computed: 1 query against all cached keys.

To predict the next word after `t3`, we only need **Row 3**:

$$\text{Output}_3 = \sum_{j=0}^{3} \underbrace{\text{Softmax}\left(\frac{\mathbf{q_3} \cdot \mathbf{k_j}}{\sqrt{d}}\right)}_{\text{Needs } q_3 \text{ and ALL } k_0 \dots k_3} \cdot \underbrace{\mathbf{v_j}}_{\text{Needs ALL } v_0 \dots v_3}$$

| Tensor            | Cached?                     | Why?                                                                                                                                                                                                                       |
| ----------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Queries ($Q$)** | **NO** _(Why not QKV?)_     | Row 3 only uses **`q3`**. Old queries (`q0, q1, q2`) only belonged to Rows 0–2, which were already computed in past steps and are **never used again**.                                                                    |
| **Keys ($K$)**    | **YES** _(Why not V-only?)_ | To know **how much weight (affinity)** token 3 gives to each past token $j$, `q3` must take a dot product with **every past Key** ($q_3 \cdot k_j$). Without $K$, you wouldn't know which past Values to pay attention to! |
| **Values ($V$)**  | **YES**                     | Once attention weights are computed from $q_3 \cdot k_j$, we take the weighted sum over **every past Value** ($v_0 \dots v_3$).                                                                                            |

### 4. Shapes During Decode (Mapping to the Self-Attention Stages)

Per layer, at decode step $t$ (MHA example: 8 heads, `head_dim = 64`):

| Self-Attention Stage | Shape |
| :--- | :--- |
| $Q$ (new token only) | `(B, 8, 1, 64)` |
| $K$, $V$ (cache + new token) | `(B, 8, t+1, 64)` |
| Stage 5: $QK^T$ | `(B, 8, 1, t+1)`: one row, no mask needed |
| Stage 6: $\times V$ | `[1, t+1] @ [t+1, 64] → [1, 64]` |

With GQA, the cache only stores the $N_{kv}$ KV heads (e.g. 2 instead of 8), which is exactly why GQA shrinks KV cache memory.

---

### 5. Why the Decode Row Is the Last Row of the Full Matrix

This uses the same worked example as [[Self-Attention]] (Prefill vs. Decode section): prompt `A B C`, then decode token `D`.

> [!info] What this diagram shows
> The 4×4 score matrix you would get **without a cache**, by re-running prefill on all 4 tokens `A B C D` (one sentence, one head, one layer). ✗ marks a cell removed by the causal mask. Compare it with the 3×3 matrix from prefill and the 1×4 row from decode.

```
             kA      kB      kC      kD
 qA  ┌   qA·kA     ✗       ✗       ✗    ┐  ◄─ same as the old prefill row A
 qB  │   qB·kA   qB·kB     ✗       ✗    │  ◄─ same as the old prefill row B
 qC  │   qC·kA   qC·kB   qC·kC     ✗    │  ◄─ same as the old prefill row C
 qD  └   qD·kA   qD·kB   qD·kC   qD·kD  ┘  ◄─ exactly the decode row
```

1. **The new column `kD` is masked out for rows A, B and C.** Their scores, softmax weights and outputs (`oA`, `oB`, `oC`) are the same numbers prefill already computed. Recomputing them is pure waste.
2. **The only new work is the bottom row.** It needs:
    - `qD`: new this step, and used only this once. This is why Q is never cached.
    - `kA … kD`: every key, to compute the scores. This is why K is cached.
    - `vA … vD`: every value, for the weighted sum. This is why V is cached.

> [!tip] Verified in code
> The Milestone 1 parity test (identical generated tokens with and without the cache) confirms that the decode row equals the last row of the full matrix.

---

### 6. Why the Causal Mask Makes Caching Safe Across All Layers

In one layer the argument above is enough. But each layer's $K$ and $V$ are computed from the **previous layer's output**, so we also need old tokens' hidden states to stay fixed in every layer.

> [!info] Diagram: dependency flow **with** the causal mask
> Follow tokens A and C down the layers (B works the same way). `oA⁰` is token A's attention output in layer 0; `hA¹` is token A's hidden state going into layer 1. The ▲ shows which tokens that attention output was allowed to see.

```
                 Layer 0                  Layer 1                  Layer 2
token A:  xA ─► kA⁰,vA⁰ ─► oA⁰ ─► hA¹ ─► kA¹,vA¹ ─► oA¹ ─► hA² ─► kA²,vA² ...
                            ▲                         ▲
                     sees only {A}             sees only {A}
                                                            ⇒ everything for A depends only on {A}

token C:  xC ─► kC⁰,vC⁰ ─► oC⁰ ─► hC¹ ─► kC¹,vC¹ ─► ...
                            ▲
                     sees {A, B, C}                         ⇒ everything for C depends only on {A, B, C}
```

When `D` arrives, the mask stops it from flowing into rows A, B or C in **any** layer. So `kA, vA, …, kC, vC` are frozen in all layers, and caching them once is safe forever.

> [!warning] Counter-example: **no** mask (bidirectional attention, like BERT)
> If old tokens were allowed to see new ones, adding `D` would change row A in layer 0, and the change would ripple upward through every layer.

```
D arrives:
  Layer 0, row A: [ qA·kA  qA·kB  qA·kC  qA·kD ]   ◄─ new entry, so oA⁰ CHANGES
        │
        ▼
  hA¹ changes  ─►  kA¹, vA¹ change  ─►  the cached layer-1 K and V are STALE
        │
        ▼
  ... and the same happens in every layer above
```

- **Layer-0 K and V would still be fine.** They come straight from the token embeddings.
- **Every layer above layer 0 goes stale.** Each new token would force a full recompute, so the cache would buy nothing.

This is why the KV cache works for causal (decoder-only) LLMs.

> **One-Sentence Rule**: _At every step $t$, the newest token uses its own **Query ($q_t$)** to search all past tokens' **Keys ($k_{0:t}$)** and blend all past tokens' **Values ($v_{0:t}$)**._