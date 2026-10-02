## Self-attention

This mechanism enables [[Transformers]] to manage long-range dependencies in data. It is fundamentally a weighting scheme that allows a model to focus on different parts of the input when producing an output. It is the engine that lets the model understand that the word `"bank"` in _"river bank"_ means something completely different than in _"money bank"_.
### Attention Calculation

For every token position $i$, attention decides how much information to pull in from the positions before it (and itself) to update that position's representation.

> **Which view do these stages show?** Stages 1–7 below show the **prefill / training** view: all $T$ tokens are processed in parallel, so there are $T$ queries and the score matrix is $T \times T$.
> During **decode** (generating one token at a time), the exact same stages run with **1 query** (the newest token) against the **cached keys and values** of all previous tokens. See [Prefill vs. Decode](#prefill-vs-decode-same-stages-different-shapes) at the end of this note and [[KV Cache]].

Imagine you are searching for a video:
- Your search term is the **Query** ($Q$).
- The video titles and tags on YouTube are the **Keys** ($K$).
- When you find a match between your Query and a Key, you watch the actual video, which is the **Value** ($V$).


![[Pasted image 20260512204645.png|480]]
#### Stage 1: The Arrival (Input Tensor)

The input tensor $X$ arrives from the previous layer (or the embedding layer).

- **Shape of $X$:** `(4, 32, 512)` ($B$, $T$, $H$) - B batch size, T - Sequence Length and H - [[Hidden Dimensions]]
- **What it represents:** 4 batches of 32 tokens, where each token is represented by a 512-dimensional continuous vector (hidden dimension).

#### Stage 2: Projection (Linear Transformation)

We pass $X$ through our three projection layers to generate the Query ($Q$), Key ($K$), and Value ($V$) matrices by multiplying it by the weight matrices.

$$ Q = XW_q, K = XW_k, V = XW_V $$

##### How are sizes of $Q,K,V$  matrices decided?
1. **The Input Dimension (Rows) is fixed by $X$:** Because we compute $X \times W$, the first dimension of $W_q, W_k, W_v$ **must** equal the model's `hidden_size` ($H = 512$). Every head receives the full 512-dimensional input $X$, and uses its own private weight matrix of shape `(512, 64)` to project all 512 features down into its own 64-dimensional subspace.
2. The Output Dimension (Columns) is chosen via number of [[Attention Head]]s and `head_dim`
	- `head_dim` is almost always 64 or 128. Conventionally it is set to $d_{\text{head}} = \frac{\text{hidden\_dimensions}}{\text{num\_attention\_heads}}$
	- **Number of heads ($N_q$ and $N_{kv}$):**
	    - **Standard Multi-Head Attention:** Every Query head has its own Key and Value head ($N_q = N_{kv} = 8$).
        - $W_q, W_k, W_v$ all have shape: $(H, N_q * head_{dim}) = (512, 8 * 64) = (512, 512)$.
    - **Grouped-Query Attention:** To reduce KV-cache VRAM, multiple Query heads share a single Key/Value head (`num_key_value_heads` < `num_attention_heads`). For example, if $N_q = 8$ and $N_{kv} = 2$:
        - $W_q$ shape: `(512, 8 * 64)` = **`(512, 512)`**
        - $W_k, W_v$ shapes: `(512, 2 * 64)` = **`(512, 128)`** _(smaller than $W_q$!)_


$$Q_1 = X W_q^{(1)}, \quad Q_2 = X W_q^{(2)}, \quad \dots, \quad Q_8 = X W_q^{(8)}$$ $$\text{where } X \text{ is } (4, 32, 512) \text{ and each } W_q^{(i)} \text{ is } (512, 64) \implies \text{each } Q_i \text{ is } (4, 32, 64)$$

- **The Math:** A matrix multiplication of $(4, 32, 512) \times (512, 512)$ for $Q$ (and for $K$, $V$ under standard MHA).
- **The Output Shapes (Standard MHA, $N_q = N_{kv} = 8$):**
    - $Q$: `(4, 32, 512)`
    - $K$: `(4, 32, 512)`
    - $V$: `(4, 32, 512)`
- **The Output Shapes (GQA example, $N_q = 8$, $N_{kv} = 2$):**
    - $Q$: `(4, 32, 512)` ($8 \times 64$)
    - $K$: `(4, 32, 128)` ($2 \times 64$)
    - $V$: `(4, 32, 128)` ($2 \times 64$)

At this point, $Q, K, V$ are still one flat vector per token (all heads packed side by side, not yet split).


**Speed Up:** 

In practice, launching 8 small matrix multiplications in a `for` loop on a GPU is very slow. So we use a **block-matrix trick**: we glue all 8 weight matrices side-by-side into one giant `W_q` matrix of shape **`(512, 512)`**:

$$W_q = \underbrace{\begin{bmatrix} \vert & \vert & & \vert \\ W_q^{(1)} & W_q^{(2)} & \cdots & W_q^{(8)} \\ \vert & \vert & & \vert \end{bmatrix}}_{(512,\; 8 \times 64) \;=\; (512,\; 512)}$$

Because of how matrix multiplication works (multiplying $X$ against each column independently), doing one big multiplication $Q = X W_q$ produces:

$$Q = X W_q = \underbrace{\begin{bmatrix} XW_q^{(1)} & XW_q^{(2)} & \cdots & XW_q^{(8)} \end{bmatrix}}_{(4,\; 32,\; 512)}$$




#### Stage 3: The Split (Multi-Head Reshaping)

This is where we divide the work among the [[Attention Head]]s. We must reshape and transpose the tensors. Assuming 8 heads with 64 dimensions. **We never split the raw input $X$. Every attention head gets to see the entire 512-dimensional input $X$.** We split the $Q, K, V$ matrices.

 > Suppose we chopped the raw input $X$ `(512)` into 8 pieces of `64` _before_ projecting:
 > - Head 1 would only ever see features `0..63` of the token.
 > - Head 2 would only ever see features `64..127` of the token.
 > If semantic meaning happened to live in feature `10` and grammar in feature `400`, no single head could ever connect them!

It isn't splitting the input $X$—it is simply **unpacking the 8 separate `(batch_size, seq_len, head_dim)` head outputs** that we computed simultaneously in Stage 2!

1. **Reshape:** We split the flat per-token vector (512) into `(num_heads, head_dim) = (8, 64)`.
    - `(4, 32, 512)` → `(4, 32, 8, 64)`, i.e. `(batch_size, seq_len, hidden)` → `(batch_size, seq_len, num_heads, head_dim)`
    - _(GQA example: $K$, $V$ go `(4, 32, 128)` → `(4, 32, 2, 64)`.)_
2. **Transpose:** We swap the sequence length ($T$) and number of heads ($N_h$) dimensions.
    - `(4, 32, 8, 64)` → `(4, 8, 32, 64)`, i.e. `(batch_size, seq_len, num_heads, head_dim)` → `(batch_size, num_heads, seq_len, head_dim)`

```
Q,K,V Vectors (512 dims)
[─────────────────────────────────────────────────────────────────]
  │         │         │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼         ▼         ▼
Head 1    Head 2    Head 3    Head 4    Head 5    Head 6    Head 7    Head 8
(64 dims) (64 dims) (64 dims) (64 dims) (64 dims) (64 dims) (64 dims) (64 dims)
```


```
[Batch, Seq_Len, Hidden] ──(Reshape)──> [Batch, Seq_Len, Heads, Head_Dim] ──(Transpose)──> [Batch, Heads, Seq_Len, Head_Dim]
```

 > Why do we transpose? PyTorch's batch matrix multiplication operates on the **last two dimensions** of a tensor. By moving the sequence length ($T$) and head dimension ($d_{\text{head}}$) to the end, we can perform attention calculations for all 8 heads and all 4 batches simultaneously using highly optimized GPU kernels.
 
 
#### Stage 4: Positional Injection (RoPE)

RoPE is applied inside each [[Attention Head]] (on each 64-dim head vector).

Before calculating attention, we must tell the model _where_ each token is in the sentence. We apply [[Rotary Positional Embeddings (RoPE)]] to $Q$ and $K$ (but **not** $V$).
1. We slice our precomputed RoPE cache to match the positions of the current tokens ($T$ positions: `0..31`):
    - `cos` and `sin` shapes: `(32, 64)`
2. We unsqueeze them to `(1, 1, 32, 64)` so they broadcast across our batch size of 4 and all heads.
3. We split our 64-dimensional $Q$ and $K$ head vectors into two 32-dimensional halves, rotate them using the sine/cosine values, and glue them back together (the `rotate_half` trick).

Now, $Q$ and $K$ contain both semantic meaning _and_ relative positional awareness.

#### Stage 5: The Matchmaking (Attention Score Calculation)

The relevance scoring step of attention multiplies **every token's query** with **the keys of all tokens**. This produces a score stating how relevant each token $j$ is to token $i$.

$$\text{Raw Scores} = \frac{QK^T}{\sqrt{D}}$$

- **The Math:** We take $Q$ of shape `(4, 8, 32, 64)` and multiply it by the transpose of $K$ of shape `(4, 8, 64, 32)`.
    - _(GQA: $K$ only has 2 heads, `(4, 2, 32, 64)`. Each KV head is shared by a group of 4 query heads, e.g. via `repeat_interleave` or `enable_gqa=True`, so the math behaves as if $K$ had 8 heads.)_
- **Scaling:** We divide by $\sqrt D$ ($\sqrt{64} = 8$) to prevent the dot products from growing too large in magnitude, which would push the softmax function into regions with dangerously small gradients.
- **Output Shape:** `(4, 8, 32, 32)` (Batch, Heads, Seq_Len, Seq_Len).
	- **`4`**: For each of the 4 sentences in the batch...
	- **`8`**: For each of the 8 attention heads...
	- **`32 x 32`**: There is a $32 \times 32$ table where cell **`[i, j]`** is the **raw score** $q_i \cdot k_j / \sqrt{D}$: how strongly token `i` matches token `j` inside that head. _(These only become percentages from `0.0` to `1.0` after Stage 5b.)_

##### Why do we calculate a separate `32 x 32` table _within each head_?
Imagine the sentence:

> _"The **trophy** didn't **fit** in the suitcase because **it** was too **big**."_

Look at the word **`"it"`** (Token $i$). To understand `"it"`, the model needs to connect `"it"` to **several different words for completely different reasons**:

- If you only had **1** `32 x 32` table and took a weighted average of all words at once, the grammar, the noun `"trophy"`, and the adjective `"big"` would all blur together into mush!
- By giving each of the **8 heads** its own separate `32 x 32` table (because each head has its own $W_q^{(h)}$ and $W_k^{(h)}$):
    - **Head 1's `32 x 32` table** (Pronoun Detective): Connects `"it"` $\rightarrow$ **`"trophy"`** (`95%` score).
    - **Head 2's `32 x 32` table** (Property Detective): Connects `"it"` $\rightarrow$ **`"big"`** (`90%` score).
    - **Head 3's `32 x 32` table** (Action Detective): Connects `"it"` $\rightarrow$ **`"fit"`** (`85%` score).

Each head gets to build its own independent `32 x 32` relationship map without interfering with the other 7 heads!

#### Stage 5b: Causal Mask + Softmax

In a decoder-only LLM, token $i$ is **only allowed to look at tokens $0 \dots i$**, never at future tokens. Before the softmax, every cell where $j > i$ (the upper triangle) is set to $-\infty$:

```
                 k0      k1      k2      k3
q0 (token 0) [ q0·k0   -inf    -inf    -inf  ]
q1 (token 1) [ q1·k0   q1·k1   -inf    -inf  ]
q2 (token 2) [ q2·k0   q2·k1   q2·k2   -inf  ]
q3 (token 3) [ q3·k0   q3·k1   q3·k2   q3·k3 ]
```

Then a softmax is applied along each row ($\text{softmax}(-\infty) = 0$), so each row becomes percentages that sum to 1:

$$\text{Attn\_Weights} = \text{Softmax}\left(\frac{QK^T}{\sqrt{D}} + M_{\text{causal}}\right)$$

- **Shape:** still `(4, 8, 32, 32)`. Now cell `[i, j]` really is "what fraction of token `i`'s attention goes to token `j`", and every cell with `j > i` is exactly `0`.
- **Why this matters for inference:** because of this mask, token $j$'s representation in **every layer** depends only on tokens $0 \dots j$. Adding a new token later never changes the rows, keys, or values of older tokens. This is exactly what makes the [[KV Cache]] valid.

![[Pasted image 20260512214442.png|560]]

#### Stage 6: Value Aggregation (Multiplying Scores by $V$)

Now that each head has its `32 x 32` probability table (`Attn_Weights`), we multiply that table by the **Value matrix $V$** of shape `(4, 8, 32, 64)`:

$$\text{Head\_Out} = \text{Attn\_Weights} \times V$$

- **The Math**: `(4, 8, 32, 32) @ (4, 8, 32, 64)` _(Look at the last two dimensions: `[32, 32] @ [32, 64] = [32, 64]`—the inner `32` cancels out!)_
- **Output Shape**: **`(4, 8, 32, 64)`** `(Batch, Heads, Seq_Len, Head_Dim)`
- **What it represents**: For each of the 32 tokens, each of the 8 heads has now produced its own **64-number summary vector**!

In other words, for each token we multiply the value vector of every other token by its attention weight and sum up the results. That weighted sum is the output of this attention step.
![[Pasted image 20260512214517.png|560]]

#### Stage 7: The Merge (Concatenating the 8 Heads & Output Projection $W_o$)

To return a tensor of the exact same shape we started with in Stage 1 (`(4, 32, 512)`), we now do the **exact reverse of Stage 3**

1. **Transpose back**: Swap the `Heads (8)` and `Seq_Len (32)` dimensions back so all 8 heads for a token sit next to each other:
    - `(4, 8, 32, 64)` $\xrightarrow{\text{transpose}}$ **`(4, 32, 8, 64)`**
2. **Concatenate (`.view`)**: Glue the 8 vectors of `64` numbers side-by-side into one single `512`-number vector per token ($8 \times 64 = 512$):
    - `(4, 32, 8, 64)` $\xrightarrow{\text{view}}$ **`(4, 32, 512)`**
```
  Head 1    Head 2    Head 3    Head 4    Head 5    Head 6    Head 7    Head 8
(64 dims) (64 dims) (64 dims) (64 dims) (64 dims) (64 dims) (64 dims) (64 dims)
  │         │         │         │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
[─────────────────────────────────────────────────────────────────]
               Concatenated Vector (8 * 64 = 512 dims) 
```
3. **Final Output Projection ($W_o$, `self.o_proj` in `attention.py`)**: Right after gluing the 8 heads side-by-side, slots `0..63` only contain Head 1's findings, slots `64..127` only contain Head 2's findings, etc. So we multiply by one final weight matrix $W_o$ of shape `(512, 512)`:
     $$\text{Output} = \text{Concat\_Heads} \times W_o \quad \longrightarrow \quad (4, 32, 512) \times (512, 512) = \mathbf{(4, 32, 512)}$$
     
This mixes the findings of all 8 heads together into the final `(4, 32, 512)` output tensor!

## Prefill vs. Decode: Same Stages, Different Shapes

Stages 1–7 above are the **prefill** view (all $T$ tokens at once). During **decode**, we feed only the 1 newest token at step $t$; its $K$, $V$ are appended to the [[KV Cache]], and the cached history is used as $K$, $V$. Nothing about the math changes; only the number of query rows does.

Using the MHA example ($N_q = N_{kv} = 8$, `head_dim = 64`):

| Stage | Prefill ($T = 32$ tokens in) | Decode at step $t$ (1 token in) |
| :--- | :--- | :--- |
| 1: $X$ | `(B, 32, 512)` | `(B, 1, 512)` |
| 3–4: $Q$ (after split + RoPE) | `(B, 8, 32, 64)` | `(B, 8, 1, 64)` |
| $K$, $V$ used for attention | `(B, 8, 32, 64)` (just computed) | `(B, 8, t+1, 64)` (cache + new token) |
| 5: $QK^T$ | `(B, 8, 32, 32)` square | `(B, 8, 1, t+1)` **one row** |
| 5b: causal mask | needed (upper triangle) | not needed: the newest token may see everything before it |
| 6: $\times V$ | `[32, 32] @ [32, 64] → [32, 64]` | `[1, t+1] @ [t+1, 64] → [1, 64]` |
| 7: merge + $W_o$ | `(B, 32, 512)` | `(B, 1, 512)` |

The decode row in Stage 5 is exactly the **last row** of the prefill matrix, which is why only $K$ and $V$ need to be cached (see [[KV Cache]]).

### Worked Example: Prompt `A B C`, Then Decode `D`

> [!info] What these diagrams show
> One sentence, **one attention head**, one layer. Each row is one token, and $d$ = `head_dim`. The batch and head dimensions are dropped so every matrix fits on screen; the real tensors repeat this exact picture for every sentence and every head.

#### Prefill: all 3 prompt tokens at once

> [!info] Diagram 1: Stage 2 (Projection)
> Each token's row of $X$ is multiplied by $W_q$, $W_k$, $W_v$. Row $i$ of $Q$, $K$, $V$ belongs to token $i$.

```
   X (3×d)          Q (3×d)     K (3×d)     V (3×d)
 ┌ xA ┐           ┌ qA ┐      ┌ kA ┐      ┌ vA ┐
 │ xB │  ──W──►   │ qB │      │ kB │      │ vB │
 └ xC ┘           └ qC ┘      └ kC ┘      └ vC ┘
```

> [!info] Diagram 2: Stages 5 + 5b (Scores and causal mask)
> $S = QK^T$. Cell `[i, j]` is how strongly token $i$'s query matches token $j$'s key. ✗ marks a masked cell (set to $-\infty$, which becomes 0 after softmax): a token can never look at a later token.

```
   Q (3×d)        Kᵀ (d×3)                  S (3×3)
 ┌ qA ┐                              ┌ qA·kA    ✗       ✗    ┐
 │ qB │  ×  [ kA  kB  kC ]   =       │ qB·kA  qB·kB     ✗    │
 └ qC ┘                              └ qC·kA  qC·kB  qC·kC   ┘
```

> [!info] Diagram 3: Stage 6 (Weighted sum of values)
> After softmax, each row of $P$ sums to 1. Row $i$ of the output $O$ is a weighted blend of the values of tokens $0 \dots i$ only.

```
     P (3×3)              V (3×d)          O (3×d)
 ┌ pAA   0    0  ┐     ┌ vA ┐     ┌ oA = pAA·vA                    ┐
 │ pBA  pBB   0  │  ×  │ vB │  =  │ oB = pBA·vA + pBB·vB           │
 └ pCA  pCB  pCC ┘     └ vC ┘     └ oC = pCA·vA + pCB·vB + pCC·vC  ┘
```

After prefill, the cache holds `K_cache = [kA, kB, kC]` and `V_cache = [vA, vB, vC]`. Only row `oC` (after the remaining layers) goes on to `lm_head` to predict `D`.

#### Decode: only the new token `D`

> [!info] Diagram 4: Decode Stages 2 + 5 (One query against all keys)
> Only `xD` is projected, giving `qD`, `kD`, `vD`; `kD` and `vD` are appended to the cache. The single query `qD` is scored against all 4 keys, giving **one row** of scores. No mask is needed, because nothing in the cache comes after `D`.

```
  Q (1×d)       Kᵀ (d×4), from the cache               S (1×4)
  [ qD ]   ×   [ kA  kB  kC  kD ]    =    [ qD·kA  qD·kB  qD·kC  qD·kD ]
```

> [!info] Diagram 5: Decode Stage 6 (Blend all cached values)
> The one row of weights blends all 4 values (3 from the cache plus the new one) into `oD`.

```
        P (1×4)                  V (4×d)           O (1×d)
[ pDA  pDB  pDC  pDD ]     ×   ┌ vA ┐
                               │ vB │    =    [ oD = pDA·vA + pDB·vB + pDC·vC + pDD·vD ]
                               │ vC │
                               └ vD ┘
```

> [!tip] Where this lives in the code
> Prefill is step 0 in `generator.py`: the whole prompt goes in and `attention.py` uses `is_causal=True`. Decode is every later step: 1 token goes in (`num_new_tokens = 1`) and `is_causal=False`.

For why Diagram 4's row is exactly the last row of the full 4×4 matrix, and why the mask makes this safe across all layers, see sections 5 and 6 of [[KV Cache]].

## Computational Complexity

 **Compute time** and **VRAM usage** scale **quadratically** with context length. If you double your context length from **512 to 1024**, the attention matrix doesn't get 2x larger; it gets **4x larger** (from 262,144 elements to 1,048,576 elements per attention head, per layer). Self-attention is one of the major contributors to any context length limitations.