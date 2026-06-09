# Rotary Position Embedding (RoPE)

# Why position encoding at all?

Attention is fundamentally a **set operation**: the output for a query is a weighted sum over the values, and permuting the key/value pairs permutes the output identically. If you feed the sequence `["the", "cat", "sat"]` or `["sat", "cat", "the"]` into the same attention head, the per-token outputs are just permuted versions of each other — the model has no way to know which token came first. A transformer without position information sees only a bag of tokens.

This is fine for pure bag-of-words models, but language is ordered: "dog bites man" and "man bites dog" have opposite meanings. We need to inject position into the architecture. The main approaches:

- **Learned absolute embeddings** (BERT, GPT-2): a lookup table of $N_{\text{max}} \times d$ added to the input. Simple, but limited to the trained max length and doesn't generalize beyond it. Also the position signal gets washed out by residual addition [residual explanation](mhc.md#standard-residual-skip-connections) throughout the layers.

- **Sinusoidal absolute embeddings** (original Transformer): This method uses fixed geometric frequencies (alternating between sin and cosine curves, for even and odd tokens in a sequence respectively), no learned parameters. This can be extrapolated to any sequence length.

- **Relative bias** (T5, ALiBi): add a position-dependent scalar to the attention logits directly: $S_{ij} = Q_i K_j^T + b_{|i-j|}$. If you do not understand this, please read [Attention](fa.md#standard-attention). This bias term creates a clean separation of content and position, but $b$ is either learned (T5, fixed max distance) or a hand-crafted decay (ALiBi). No modification to Q, K, or V themselves.

## Absolute vs Relative position encoding
The absolute encoding examples we covered above don't naturally express which token is *relative* to which, just that each token has a unique position. The model has to learn to decode relative position on it's own from the absolute position we encoded. In simple terms, in the sentence "john likes mary", the model only knows "john is at position 0, likes is at position 1, mary is at position 2", but it needs to learn on it's own that "john is 1 token to the left of likes, and 2 tokens to the left of mary". This is a harder learning problem than directly encoding relative position.

- **Rotary (RoPE)**: the approach we cover here. Instead of adding position to the input or biasing the scores, RoPE rotates the Q and K vectors so that the dot product $Q_i K_j^T$ inherently expresses the *relative* position $i-j$ through the rotation angle difference.

# The core idea

We want position-encoded query and key functions $f_Q(q_m, m)$, $f_K(k_n, n)$ such that:

$$
f_Q(q_m, m)^T f_K(k_n, n) = g(q_m, k_n, m-n)
$$

This means, that we want functions $f_Q$ and $f_K$ that take in the token embedding and the position, and produce a new vector such that when we take the dot product of the query and key, the resulting transformation only on the relative position $(m-n)$, not on the absolute positions $m$ and $n$.

RoPE achieves this by applying a rotation matrix $R(p) \in \mathbb{R}^{d \times d}$ to the vector at position $p$:

$$
f_Q(q_m, m) = R(m) q_m, \qquad f_K(k_n, n) = R(n) k_n
$$

# Mathematical definition

$R(p)$ is [block-diagonal](https://en.wikipedia.org/wiki/Block_matrix#Block_diagonal_matrices) with $2 \times 2$ rotation blocks. In simple terms, $R(p)$ has $2\times 2$ matrices $R_j(p)$, arranged along it's diagonal.


$$
R(p)
=
\begin{bmatrix}
R_0(p) & 0 & \cdots & 0 \\
0 & R_1(p) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & R_{\frac{d}{2}-1}(p)
\end{bmatrix}
\qquad
R_j(p) = \begin{bmatrix}
\cos(p \theta_j) & -\sin(p \theta_j) \\
\sin(p \theta_j) & \cos(p \theta_j)
\end{bmatrix}
$$

that is,

$$
R(p) =
\begin{bmatrix}
\cos(p\theta_0) & -\sin(p\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
\sin(p\theta_0) & \cos(p\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(p\theta_1) & -\sin(p\theta_1) & \cdots & 0 & 0 \\
0 & 0 & \sin(p\theta_1) & \cos(p\theta_1) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos\!\left(p\theta_{\frac{d}{2}-1}\right) &
-\sin\!\left(p\theta_{\frac{d}{2}-1}\right) \\
0 & 0 & 0 & 0 & \cdots &
\sin\!\left(p\theta_{\frac{d}{2}-1}\right) &
\cos\!\left(p\theta_{\frac{d}{2}-1}\right)
\end{bmatrix}
$$

there are $d/2$ pairs of dimensions, each pair $(2j, 2j+1)$ is rotated in a 2D plane by an angle that depends on the position $p$ and the frequency $\theta_j$.

Let $\text{base}$ be a large constant (default: $10000$), and for $j = 0, 1, \dots, d/2 - 1$ define the angular frequencies:

$$
\theta_j = \text{base}^{-2j/d}
$$



Applying $R(p)$ to a vector $x \in \mathbb{R}^d$ gives, for each pair $j$:

$$
\begin{aligned}
x'_{2j}   &= x_{2j} \cos(p \theta_j) - x_{2j+1} \sin(p \theta_j) \\
x'_{2j+1} &= x_{2j} \sin(p \theta_j) + x_{2j+1} \cos(p \theta_j)
\end{aligned}
$$

## Components

- $d$: head dimension (must be even).
- $p$: token position index (0-indexed).
- $\text{base}$: frequency base constant (default $10000$). Higher base gives lower frequencies and longer effective context.
- $\theta_j$: angular frequency for dimension pair $j$. Decays geometrically: $\theta_0 = 1 \text{ rad}$, $\theta_{d/2-1} = \text{base}^{-(d-2)/d}$.
- $R_j(p)$: $2 \times 2$ rotation matrix for pair $j$ at position $p$.
- $R(p)$: full $d \times d$ block-diagonal rotation matrix.

# Frequency decay

Low $j$ pairs (high frequency) rotate quickly per position step: they resolve near-neighbor position differences. High $j$ pairs (low frequency) rotate slowly — they encode long-range offsets. The geometric progression gives a multi-scale position representation: nearby tokens align on high-frequency pairs while distant tokens only stay aligned on low-frequency pairs. This structure is analogous to the sinusoidal position encoding in the original Transformer.

# The relative property of rotation matrices

Rotation matrices form a group: $R_j(m)^T R_j(n) = R_j(n-m)$ for each $2 \times 2$ block. Since $R(p)$ is block-diagonal, the same holds for the full matrix:

$$
R(m)^T R(n) = R(n-m)
$$

Substituting into the attention dot product:

$$
(R(m) q_m)^T (R(n) k_n) = q_m^T R(m)^T R(n) k_n = q_m^T R(n-m) k_n
$$

The result depends only on the relative offset $(n-m)$, not on $m$ and $n$ individually. The $2 \times 2$ block structure means this holds for each pair independently — no cross-dimensional mixing.

In simpler terms, in the massive matrix multiplication of $Q \cdot K^T$, when token $m$ from Q attends to token $n$ from K, the resultant matrix automatically encodes the relative position $n-m$, through this property. This applies to all pairs of tokens, so the model now has true relative position information baked into the attention scores, without needing to learn it from absolute positions.

# Why

- **Relative position encoding:** the score decays with token distance, which is the correct inductive bias for natural language. Generalizes to unseen sequence lengths better than absolute position embeddings.
- **No learned parameters:** the frequencies $\theta_j$ are fixed hyper-parameters. No position lookup table, no extra weights to save or load.
- **Natural distance decay:** on average, $q_m^T R(n-m) k_n$ decreases as $|n-m|$ grows because the rotated vectors become progressively misaligned. This emerges from the geometry of rotation, not from learned weights.
- **Compatible with FA2 tiling:** RoPE is applied per-tile after loading $Q_i$ and $K_j$ into SRAM, before computing $S_{ij}$. The rotation is element-wise and adds negligible overhead.
- **Light on KV cache:** $\cos$/$\sin$ values are precomputed once per context and reused across all layers and heads. The KV cache stores only raw $K$ (or compressed $K$ in architectures like CSA/HCA), not position-specific variants.
- **No additive interaction:** unlike ahanddditive position embeddings, RoPE does not mix with the token embedding through addition. It operates purely on the $QK^T$ dot product, making the position signal orthogonal to the semantic content.

# Why V isn't rotated

RoPE only rotates Q and K. V is left untouched. This is a common point of confusion, so it's worth spelling out explicitly.

The attention output is:
$$
O = \text{softmax}(Q K^T) V
$$

The position information enters through $Q K^T$ — the attention weights $P = \text{softmax}(Q K^T)$ determine *which* tokens to attend to and *how much*. The value $V$ only provides the *content* that gets aggregated according to those weights. Rotating V would have no effect on the attention distribution (it's not in the $Q K^T$ term), and it would rotate the output vectors into a different coordinate frame that then needs to be un-rotated by the downstream projection — extra compute for zero benefit.

Concretely:
$$
\text{softmax}(Q K^T) \, (R V) \neq R \cdot \big(\text{softmax}(Q K^T) V\big)
$$

If you rotate V (but not the output projection), the weighted sum is computed in a rotated space and then projected back by the output $W_O$, which the network would have to learn to undo. Since the relative position is already fully encoded by the pair $(R(m) q_m, R(n) k_n)$, rotating V is redundant — it adds a learnable correction for a problem that doesn't exist.

The same reasoning applies to **causal masking** and **Flash Attention tiling** — causal masks only filter $S_{ij}$ (the QK score tile), and FA2's inner loop only touches Q and K for the position-dependent rotation. V flows through unchanged.

# Integration with Flash Attention 2

In the tiled FA2 loop (see [fa.md](./fa.md)), RoPE is applied immediately after loading each $Q_i$ and $K_j$ tile from HBM into SRAM:

1. Load $Q_i$ tile from HBM.
2. For each row $Q_i[r]$ at global position $p = m_i + r$, compute $R(p)\, Q_i[r]$.
3. Load $K_j$ tile from HBM.
4. For each row $K_j[r]$ at global position $p = n_j + r$, compute $R(p)\, K_j[r]$.
5. Compute $S_{ij} = Q_i K_j^T$ and proceed with the online softmax.

The $\cos$ and $\sin$ values for all positions $0 \dots N-1$ are precomputed as a lookup table of shape $N \times d$ (shared across all layers and heads). The rotation uses two element-wise multiply-adds per loaded dimension — effectively free compared to the $B_r \times B_c$ matmul in $S_{ij}$.

# Worked example ($d = 4$, $\text{base} = 100$)

Let $d = 4$ ($2$ pairs), $\text{base} = 100$, position $p = 3$:

$$
\theta_0 = 100^{-0/4} = 1, \qquad \theta_1 = 100^{-2/4} = 100^{-0.5} = 0.1
$$

For $x = [1, 2, 3, 4]^T$:

$$
\begin{aligned}
x'_0 &= 1 \cdot \cos(3) - 2 \cdot \sin(3) \approx -0.990 - 2(0.141) = -1.272 \\
x'_1 &= 1 \cdot \sin(3) + 2 \cdot \cos(3) \approx 0.141 + 2(-0.990) = -1.839 \\[4pt]
x'_2 &= 3 \cdot \cos(0.3) - 4 \cdot \sin(0.3) \approx 3(0.955) - 4(0.296) = 1.681 \\
x'_3 &= 3 \cdot \sin(0.3) + 4 \cdot \cos(0.3) \approx 3(0.296) + 4(0.955) = 4.708
\end{aligned}
$$

Pair $0$ (high frequency $\theta_0 = 1$) rotated significantly; pair $1$ (low frequency $\theta_1 = 0.1$) barely moved. Repeating at $p = 4$ would rotate pair $0$ by another full radian while pair $1$ only changes by $0.1$ rad — pair $0$ encodes fine-grained position, pair $1$ encodes coarse position.

Notice that each rotation, only changes the direction of the vector in that 2D plane, not the magnitude. This is because, by definition, rotation matrices are orthonormal.

# Quick reference

- RoPE applies a block-diagonal rotation $R(p) \in \mathbb{R}^{d \times d}$ to each Q and K vector based on its position $p$.
- Frequencies: $\theta_j = \text{base}^{-2j/d}$ for $j = 0, \dots, d/2-1$, with $\text{base} = 10000$ by default.
- Key identity: $(R(m) q)^T (R(n) k) = q^T R(n-m) k$.
- Zero extra parameters per layer. Precomputed $\cos$/$\sin$ lookup table of size $N \times d$, shared across all layers.
- Applied tile-by-tile in FA2, fused into the HBM $\to$ SRAM load. Cost: $2$ MAdds per element.
- $d$ must be even; typical values: $d = 64, 128, 256$.
- Extensions for longer context (no re-training):
  - **NTK-aware scaling:** increase base (e.g. $500000+$) to stretch frequencies.
  - **YaRN:** interpolate frequencies with a ramp to preserve high-frequency resolution.
  - **Position Interpolation (PI):** linearly scale position indices by $\frac{L_\text{train}}{L_\text{infer}}$.
