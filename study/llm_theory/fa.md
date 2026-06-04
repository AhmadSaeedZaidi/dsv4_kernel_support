# Flash Attention 2

Replaces the standard attention $O = \text{softmax}(QK^T)V$ (which materializes a full $N \times N$ attention matrix in HBM) with an IO-aware tiled algorithm that streams Q, K, V through on-chip SRAM and never writes the $N \times N$ matrix out.

Motivation: standard attention is bottlenecked by memory, not FLOPs. For a sequence of length $N$ with head dim $d$, the $N \times N$ softmax matrix costs $O(N^2)$ memory and forces $O(N^2)$ HBM reads/writes on GPU. Flash Attention 2 cuts this to $O(N)$ HBM accesses (linear in sequence length) by recomputing attention in tiles and using the online softmax trick to merge partial results.

# Standard Attention 

$$
S = QK^T \in R^{N \times N}, \qquad P = Softmax_{row}(S) \in R^{N \times N}, \qquad O = PV \in R^{N \times d}
$$

## Components

- $Q, K, V \in R^{N \times d}$: query, key, value matrices for a single attention head, where $N$ is sequence length and $d$ is head dimension.
- $S \in R^{N \times N}$: attention scores, one scalar per (query, key) pair.
- $P \in R^{N \times N}$: attention weights after row-wise softmax. Each row sums to 1.
- $O \in R^{N \times d}$: output, one $d$-dim vector per query.

## Why this is inefficient

- **Quadratic Memory Scaling $O(N^2)$:** $S$ and $P$ are $N \times N$. For $N = 8192$ and fp16 that's $8192^2 \cdot 2\text{B} \approx 128\text{ MB}$ per head, per layer. With 32 layers and 32 heads, this is already enormous amounts of memory. Since N is our sequence length, our memory scales quadratically in longer conversations, which is the main bottleneck for long-context LLMs.

- **Bandwidth bound $O(N^2)$:** the GPU makes 3 trips to HBM.
    1. Read $Q, K$ to compute $S$. Write $S$ back to HBM.
    2. Read $S$ from HBM to compute the row-max for softmax, then write $P$ back.
    3. Read $P$ again to multiply with $V$. Three full passes over an $N \times N$ matrix that mostly lives in slow HBM, not fast SRAM.

    HBM might higher bandwidth than VRAM, but it's still nothing compared to on-chip SRAM. We want to minimize HBM trips.

# Flash Attention 2:

Break down the Q, K, V matrices into tiles that fit in SRAM, and compute attention block by block.

for now, we will assume a tile has size $m_r \times m_c$, where $m_r$ is the number of rows of Q, V and $m_c$ is the number of rows of $K^T$. This size is dynamically computed based on SRAM in the hardware, during compilation.

The full forward pass for a single Q block $Q_i$ loops over all K, V blocks $(K_j, V_j)$ where $K_j$ and $V_j$ are matrices of size $B_c \times d$, $B_r \times d$ respectively and accumulates an output $O_i$ using an online softmax. Concretely:

Inside the inner loop (for a fixed Query block $i$ and streaming Key/Value block $j$):

1. Compute raw scores for the tile:
$$S_{ij} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$$

2. Find the local row maximums and compute stable exponentials:
$$m_{ij} = \text{rowmax}(S_{ij}) \in \mathbb{R}^{B_r}$$
$$P_{ij} = e^{S_{ij} - m_{ij}} \in \mathbb{R}^{B_r \times B_c}$$

3. Compute the local row sums of the exponentials:
$$\ell_{ij} = \text{rowsum}(P_{ij}) \in \mathbb{R}^{B_r}$$

4. Update the global running statistics (Max and Denominator) using the correction factor:
$$m_i^{\text{new}} = \max(m_i^{\text{old}}, m_{ij})$$
$$\ell_i^{\text{new}} = \left( e^{m_i^{\text{old}} - m_i^{\text{new}}} \odot \ell_i^{\text{old}} \right) + \left( e^{m_{ij} - m_i^{\text{new}}} \odot \ell_{ij} \right)$$

5. Update the running, unnormalized Output accumulator ($\tilde{O}_i$):
$$\tilde{O}_i^{\text{new}} = \left( e^{m_i^{\text{old}} - m_i^{\text{new}}} \odot \tilde{O}_i^{\text{old}} \right) + \left( e^{m_{ij} - m_i^{\text{new}}} \odot (P_{ij} V_j) \right) \in \mathbb{R}^{B_r \times d}$$

---
After the inner loop finishes (all Key/Value blocks $j$ have been processed):

6. Perform the final Safe Normalization to get the true Attention Output ($O_i$):
$$O_i = \tilde{O}_i^{\text{final}} \oslash \ell_i^{\text{final}}$$

after each inner step, $O_i$ is rescaled to compensate for the running max update. This is an [element-wise division](../../notation.md#element-wise-division). 

## Components

- $Q_i \in R^{B_r \times d}$: tile $i$ of the query matrix, of block size $B_r$ rows. $i = 1 \dots T_r$ where $T_r = \lceil N / B_r \rceil$.
- $K_j, V_j \in R^{B_c \times d}$: tile $j$ of the key and value matrices, of block size $B_c$ rows. $j = 1 \dots T_c$ where $T_c = \lceil N / B_c \rceil$.
- $S_{ij} \in R^{B_r \times B_c}$: block of attention scores between $Q_i$ and $K_j$. Lives only in SRAM, never written to HBM.
- $m_i \in R^{B_r}$: running row-max for $Q_i$ across all $K$ blocks seen so far. Initialized to $-\infty$ (or just the first block's rowmax, same thing).
- $\ell_i \in R^{B_r}$: running row-sum of unnormalized exp scores for $Q_i$. Initialized to $0$.
- $O_i \in R^{B_r \times d}$: running output for $Q_i$. Initialized to $0$. After all $T_c$ inner steps it is divided by $\ell_i$ to give the true softmax-weighted output.
- $\text{rowmax}(\cdot), \text{rowsum}(\cdot)$: per-row max and sum. Cheap, on chip.
- $B_r, B_c$: block sizes. Typical: $B_r = 64$ or $128$, $B_c = 64$ or $128$. Picked so that $Q_i, K_j, V_j, S_{ij}, P_{ij}$ all fit in SRAM together.

## Why

- **Tiling $B_r, B_c$:** instead of computing the full $N \times N$ softmax matrix, we only ever materialize a $B_r \times B_c$ block in fast on-chip SRAM. Once the inner step is done, $S_{ij}$ and $P_{ij}$ are discarded. HBM only ever sees the $N \times d$ inputs and outputs.
- **Online softmax (Milakov & Gimelshein 2018):** the standard softmax needs the full row to know the max. The online trick fuses a new $K$ block into the running $(m_i, \ell_i, O_i)$ by rescaling the previous accumulator with $e^{m_i - m_i^{new}}$. This is the same identity you'd get if you had computed the full row at once, just stitched together block by block. Read more here: [online softmax](https://arxiv.org/abs/1805.02867).
- **No HBM round trip:** standard attention reads/writes $S$ and $P$ three times. Flash Attention 2 only loads $Q_i, K_j, V_j$ once each, does all the math in SRAM, and writes the final $O_i$ back. Total HBM traffic is $O(N d)$ instead of $O(N^2 d)$.
- **Recomputation in backward pass:** normally the backward pass would need the saved $S$ and $P$ to compute gradients w.r.t. $Q, K, V$. Flash Attention 2 throws them away and re-tiles the forward on the fly during backward. Recomputing is cheap (it's the same flops we were already doing) and saves the $O(N^2)$ memory that would have been needed to store $S, P$.
- **FLOPs are unchanged:** Flash Attention 2 is exactly the same math as standard attention, just rearranged. The speedup is purely from memory access patterns, not from fewer operations.

## Worked Example

since this is a hard concept, I will walk through a tiny example with $N = 6$, $d = 2$, $B_r = 2$, $B_c = 2$. That gives $T_r = T_c = 3$ tiles of $2$ rows each. I will fix a single query tile $Q_1$ (rows $1$--$2$ of $Q$) and walk all three inner steps $j = 1, 2, 3$, then do the final step-$6$ normalization. The arithmetic below is also reproduced and verified by [`verify_fa_example.py`](./verify_fa_example.py) in this directory.

We start with a row-$Q$, a transposed $K^T$ (so the $\text{row} \times \text{col}$ shapes line up with the algorithm on line $31$), and a row-$V$:
$$
Q = \begin{bmatrix}
 2 & -1 \\
-2 &  5 \\
 4 &  1 \\
 0 &  2 \\
 3 &  0 \\
 1 & -3
\end{bmatrix}
\qquad
K^T = \begin{bmatrix}
 1 &  0 &  4 & -1 &  5 &  2 \\
-2 &  3 &  1 &  2 &  0 & -4
\end{bmatrix}
\qquad
V = \begin{bmatrix}
 3 &  0 \\
 1 &  2 \\
-2 &  1 \\
 5 & -3 \\
 4 &  0 \\
-1 &  2
\end{bmatrix}
$$
(so $K = (K^T)^\top$ is $6 \times 2$, e.g. the first two rows of $K$ are $[1,-2]$ and $[0, 3]$, matching $K^T$'s first two columns).

### Initialization

The running statistics are vectors of length $B_r = 2$ (one entry per row of $Q_1$), and the running output $O_1$ is a $B_r \times d$ matrix. All start at the "empty product" / "empty sum" identities for softmax:
$$
m_0 = \begin{bmatrix} -\infty \\ -\infty \end{bmatrix}, \qquad
\ell_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \qquad
O_0 = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
$$

### Tiles for $Q_1$

Row-tile of $Q$ and the three column-tiles of $K^T$ (i.e. the three row-tiles of $K$):
$$
Q_1 = \begin{bmatrix} 2 & -1 \\ -2 & 5 \end{bmatrix}, \qquad
K_1^T = \begin{bmatrix} 1 & 0 \\ -2 & 3 \end{bmatrix}, \quad
K_2^T = \begin{bmatrix} 4 & -1 \\ 1 & 2 \end{bmatrix}, \quad
K_3^T = \begin{bmatrix} 5 & 2 \\ 0 & -4 \end{bmatrix}
$$
$$
V_1 = \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix}, \quad
V_2 = \begin{bmatrix} -2 & 1 \\ 5 & -3 \end{bmatrix}, \quad
V_3 = \begin{bmatrix} 4 & 0 \\ -1 & 2 \end{bmatrix}
$$

### Inner step $(i = 1,\ j = 1)$: tile $(Q_1, K_1, V_1)$

**Step 1.** $S_{11} = Q_1 K_1^T \in \mathbb{R}^{B_r \times B_c}$:
$$
S_{11}
= \begin{bmatrix} 2 & -1 \\ -2 & 5 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ -2 & 3 \end{bmatrix}
= \begin{bmatrix} 2\cdot 1 + (-1)\cdot(-2) & 2\cdot 0 + (-1)\cdot 3 \\ -2\cdot 1 + 5\cdot(-2) & -2\cdot 0 + 5\cdot 3 \end{bmatrix}
= \begin{bmatrix} 4 & -3 \\ -12 & 15 \end{bmatrix}
$$

**Step 2.** $m_{11} = \text{rowmax}(S_{11})$, $P_{11} = e^{S_{11} - m_{11}}$ (subtracting the row max for numerical stability, the standard softmax trick):
$$
m_{11} = \begin{bmatrix} 4 \\ 15 \end{bmatrix}, \qquad
P_{11} = \begin{bmatrix} e^{4-4} & e^{-3-4} \\ e^{-12-15} & e^{15-15} \end{bmatrix} = \begin{bmatrix} 1 & e^{-7} \\ e^{-27} & 1 \end{bmatrix}
$$

**Step 3.** $\ell_{11} = \text{rowsum}(P_{11})$:
$$
\ell_{11} = \begin{bmatrix} 1 + e^{-7} \\ e^{-27} + 1 \end{bmatrix}
$$

**Step 4.** Update running $m_1, \ell_1$. Since $m_0 = -\infty$ the old terms vanish, and $m_1^{\text{new}}$ just inherits the new tile's max:
$$
m_1^{\text{new}} = \max(m_0, m_{11}) = \begin{bmatrix} 4 \\ 15 \end{bmatrix}, \qquad
e^{m_0 - m_1^{\text{new}}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \qquad
e^{m_{11} - m_1^{\text{new}}} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$
$$
\ell_1^{\text{new}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \odot \ell_0 + \begin{bmatrix} 1 \\ 1 \end{bmatrix} \odot \ell_{11} = \ell_{11} = \begin{bmatrix} 1 + e^{-7} \\ 1 + e^{-27} \end{bmatrix}
$$

**Step 5.** Update running $O_1$. Same story: $O_0 = 0$, so only the $P_{11} V_1$ term survives:
$$
P_{11} V_1 = \begin{bmatrix} 1 & e^{-7} \\ e^{-27} & 1 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 3 + e^{-7} & 2e^{-7} \\ 3e^{-27} + 1 & 2 \end{bmatrix} = O_1^{\text{new}}
$$

### Inner step $(i = 1,\ j = 2)$: tile $(Q_1, K_2, V_2)$

**Step 1.**
$$
S_{12} = Q_1 K_2^T = \begin{bmatrix} 2 & -1 \\ -2 & 5 \end{bmatrix} \begin{bmatrix} 4 & -1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 7 & -4 \\ -3 & 12 \end{bmatrix}
$$

**Step 2.**
$$
m_{12} = \begin{bmatrix} 7 \\ 12 \end{bmatrix}, \qquad
P_{12} = \begin{bmatrix} 1 & e^{-11} \\ e^{-15} & 1 \end{bmatrix}
$$

**Step 3.**
$$
\ell_{12} = \begin{bmatrix} 1 + e^{-11} \\ 1 + e^{-15} \end{bmatrix}
$$

**Step 4.** *This is the first time the online-softmax correction actually fires* — row $0$'s running max moves from $4$ to $7$, so the old $O_1$ has to be rescaled by $e^{4 - 7} = e^{-3}$ before the new $P_{12} V_2$ term is added:
$$
m_1^{\text{new}} = \max\!\left( \begin{bmatrix} 4 \\ 15 \end{bmatrix},\ \begin{bmatrix} 7 \\ 12 \end{bmatrix} \right) = \begin{bmatrix} 7 \\ 15 \end{bmatrix}
$$
$$
e^{m_1^{\text{old}} - m_1^{\text{new}}} = \begin{bmatrix} e^{4-7} \\ e^{15-15} \end{bmatrix} = \begin{bmatrix} e^{-3} \\ 1 \end{bmatrix}, \qquad
e^{m_{12} - m_1^{\text{new}}} = \begin{bmatrix} e^{7-7} \\ e^{12-15} \end{bmatrix} = \begin{bmatrix} 1 \\ e^{-3} \end{bmatrix}
$$
$$
\ell_1^{\text{new}} = \begin{bmatrix} e^{-3} \\ 1 \end{bmatrix} \odot \ell_1^{\text{old}} + \begin{bmatrix} 1 \\ e^{-3} \end{bmatrix} \odot \ell_{12}
= \begin{bmatrix} e^{-3}(1 + e^{-7}) + (1 + e^{-11}) \\ 1\cdot(1 + e^{-27}) + e^{-3}(1 + e^{-15}) \end{bmatrix}
= \begin{bmatrix} 1 + e^{-3} + e^{-10} + e^{-11} \\ 1 + e^{-3} + e^{-18} + e^{-27} \end{bmatrix}
$$

**Step 5.**
$$
P_{12} V_2 = \begin{bmatrix} 1 & e^{-11} \\ e^{-15} & 1 \end{bmatrix} \begin{bmatrix} -2 & 1 \\ 5 & -3 \end{bmatrix} = \begin{bmatrix} -2 + 5e^{-11} & 1 - 3e^{-11} \\ 5 - 2e^{-15} & -3 + e^{-15} \end{bmatrix}
$$
$$
O_1^{\text{new}} = \begin{bmatrix} e^{-3} \\ 1 \end{bmatrix} \odot O_1^{\text{old}} + \begin{bmatrix} 1 \\ e^{-3} \end{bmatrix} \odot (P_{12} V_2)
= \begin{bmatrix} 3e^{-3} + e^{-10} - 2 + 5e^{-11} & 2e^{-10} + 1 - 3e^{-11} \\ 1 + 3e^{-27} + 5e^{-3} - 2e^{-18} & 2 - 3e^{-3} + e^{-18} \end{bmatrix}
$$

### Inner step $(i = 1,\ j = 3)$: tile $(Q_1, K_3, V_3)$

**Step 1.**
$$
S_{13} = Q_1 K_3^T = \begin{bmatrix} 2 & -1 \\ -2 & 5 \end{bmatrix} \begin{bmatrix} 5 & 2 \\ 0 & -4 \end{bmatrix} = \begin{bmatrix} 10 & 8 \\ -10 & -24 \end{bmatrix}
$$

**Step 2.**
$$
m_{13} = \begin{bmatrix} 10 \\ 8 \end{bmatrix}, \qquad
P_{13} = \begin{bmatrix} 1 & e^{-2} \\ e^{-18} & e^{-32} \end{bmatrix}
$$

**Step 3.**
$$
\ell_{13} = \begin{bmatrix} 1 + e^{-2} \\ e^{-18} + e^{-32} \end{bmatrix}
$$

**Step 4.** Row $0$'s running max moves again, from $7$ to $10$:
$$
m_1^{\text{new}} = \max\!\left( \begin{bmatrix} 7 \\ 15 \end{bmatrix},\ \begin{bmatrix} 10 \\ 8 \end{bmatrix} \right) = \begin{bmatrix} 10 \\ 15 \end{bmatrix}
$$
$$
e^{m_1^{\text{old}} - m_1^{\text{new}}} = \begin{bmatrix} e^{7-10} \\ e^{15-15} \end{bmatrix} = \begin{bmatrix} e^{-3} \\ 1 \end{bmatrix}, \qquad
e^{m_{13} - m_1^{\text{new}}} = \begin{bmatrix} e^{10-10} \\ e^{8-15} \end{bmatrix} = \begin{bmatrix} 1 \\ e^{-7} \end{bmatrix}
$$
$$
\ell_1^{\text{new}}
= \begin{bmatrix} e^{-3} \\ 1 \end{bmatrix} \odot \ell_1^{\text{old}} + \begin{bmatrix} 1 \\ e^{-7} \end{bmatrix} \odot \ell_{13}
= \begin{bmatrix}
 1 + e^{-2} + e^{-3} + e^{-6} + e^{-13} + e^{-14} \\
 1 + e^{-3} + e^{-18} + e^{-25} + e^{-27} + e^{-39}
\end{bmatrix}
\approx \begin{bmatrix} 1.1876 \\ 1.0498 \end{bmatrix}
$$

**Step 5.**
$$
P_{13} V_3 = \begin{bmatrix} 1 & e^{-2} \\ e^{-18} & e^{-32} \end{bmatrix} \begin{bmatrix} 4 & 0 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 4 - e^{-2} & 2e^{-2} \\ 4e^{-18} - e^{-32} & 2e^{-32} \end{bmatrix}
$$
$$
O_1^{\text{new}} = \begin{bmatrix} e^{-3} \\ 1 \end{bmatrix} \odot O_1^{\text{old}} + \begin{bmatrix} 1 \\ e^{-7} \end{bmatrix} \odot (P_{13} V_3)
= \begin{bmatrix}
 4 - e^{-2} - 2e^{-3} + 3e^{-6} + e^{-13} + 5e^{-14} & 2e^{-2} + e^{-3} + 2e^{-13} - 3e^{-14} \\
 1 + 5e^{-3} - 2e^{-18} + 3e^{-27} + 4e^{-25} - e^{-39} & 2 - 3e^{-3} + e^{-18} + 2e^{-39}
\end{bmatrix}
\approx \begin{bmatrix} 3.768 & 0.320 \\ 1.249 & 1.851 \end{bmatrix}
$$

After all $T_c = 3$ inner steps we have
$$
m_1^{\text{final}} = \begin{bmatrix} 10 \\ 15 \end{bmatrix}, \qquad
\ell_1^{\text{final}} = \begin{bmatrix} 1.1876 \\ 1.0498 \end{bmatrix}, \qquad
\tilde{O}_1^{\text{final}} \approx \begin{bmatrix} 3.768 & 0.320 \\ 1.249 & 1.851 \end{bmatrix}
$$

### Step 6: final safe normalization (and what $\text{diag}(\cdot)$ actually means here)

The output $\tilde{O}_1^{\text{final}}$ is still the *unnormalized* weighted sum of $V$ rows. Each row was scaled by $e^{S - m_1^{\text{final}}}$, so to recover the true softmax-weighted average we divide row $k$ by $\ell_1^{\text{final}}[k]$. The "$\text{diag}(\ell_1)^{-1}$" notation is just a tidy way to write this row-scaling as a left-multiplication by a diagonal matrix:

$$
\text{diag}(\ell_1^{\text{final}}) = \begin{bmatrix} \ell_1^{\text{final}}[0] & 0 \\ 0 & \ell_1^{\text{final}}[1] \end{bmatrix} = \begin{bmatrix} 1.1876 & 0 \\ 0 & 1.0498 \end{bmatrix}, \qquad
\text{diag}(\ell_1^{\text{final}})^{-1} = \begin{bmatrix} 1/1.1876 & 0 \\ 0 & 1/1.0498 \end{bmatrix} \approx \begin{bmatrix} 0.8420 & 0 \\ 0 & 0.9526 \end{bmatrix}
$$

This matrix only has entries on the diagonal, so multiplying it by $\tilde{O}_1^{\text{final}}$ just rescales each row independently — row $0$ gets divided by $\ell_1[0]$, row $1$ gets divided by $\ell_1[1]$. That is exactly the element-wise division $\tilde{O}_1 \oslash \ell_1$ on line $62$, written as a matrix product:

$$
O_1 = \text{diag}(\ell_1^{\text{final}})^{-1} \cdot \tilde{O}_1^{\text{final}}
\approx \begin{bmatrix} 0.8420 & 0 \\ 0 & 0.9526 \end{bmatrix} \begin{bmatrix} 3.768 & 0.320 \\ 1.249 & 1.851 \end{bmatrix}
\approx \begin{bmatrix} 3.768 / 1.1876 & 0.320 / 1.1876 \\ 1.249 / 1.0498 & 1.851 / 1.0498 \end{bmatrix}
\approx \begin{bmatrix} 3.173 & 0.270 \\ 1.190 & 1.763 \end{bmatrix}
$$

**Sanity check.** Computing $\text{softmax}(Q_1 K^T) V$ the standard (non-tiled) way on the same matrices gives the same $O_1$ to machine precision (max abs diff $\approx 2.2 \times 10^{-16}$ in the script) — the algorithm is exact, only the memory access pattern changed.

# Work partitioning (FA2 vs FA1)

FA1 parallelized by also splitting the inner loop over $K$ blocks across thread blocks, then reducing. This meant multiple thread blocks were redundantly touching the same $Q_i$ tile and writing partial outputs to HBM, which the reduction then had to merge.

FA2 fixes this: each thread block is assigned a slice of $Q$ (a few $Q_i$ tiles) and is responsible for the full inner loop over $K, V$. The result is written directly to HBM with no inter-block reduction.

## Why

- **No redundant HBM writes:** in FA1, two thread blocks working on overlapping $Q_i$ both wrote to HBM. FA2 assigns each query row to exactly one thread block, so each output is written once.
- **Better arithmetic intensity per thread block:** FA2 thread blocks do more compute per byte loaded from HBM (the same $K_j, V_j$ is reused across many $Q_i$ inside the block). The GPU is finally compute-bound instead of memory-bound.
- **Cross-block communication gone:** no more atomic adds, no more partial buffers, no more reduction kernel. The output $O$ comes out ready in HBM.
- **Same SRAM budget, more useful work:** with the saved HBM bandwidth, FA2 can afford to keep $Q_i, K_j, V_j$ resident longer and overlap softmax epilogue with the matmul. This is the second key win.

# Causal masking

For decoder attention, we don't want query $i$ attending to key $j > i$. There are two ways to handle this.

**Option A: skip masked blocks.** in the inner loop, if the entire $K_j$ block lies in the future, skip it. If it straddles the diagonal, compute it but mask out the upper triangle:

$$
S_{ij} = Q_i K_j^T + M_{ij}, \qquad M_{ij} \in \{0, -\infty\}^{B_r \times B_c}
$$

then proceed with the same online softmax. The $-\infty$ entries become $0$ after exp, so they contribute nothing to $\ell_i$ or $O_i$.

**Option B: no mask needed.** in the strict causal case, you can schedule $Q_i$ tiles so that the inner loop only iterates over $K_j$ with $j \le i$ (i.e. the $K, V$ blocks that are already "in the past"). No mask, no wasted compute, just a careful loop bound.

## Why

- **Option A is general:** handles arbitrary masks (sliding window, document boundaries, prefix LM, etc.). The mask is just a $0 / -\infty$ add before softmax, doesn't change the algorithm.
- **Option B is fast for pure causal:** the FA2 paper uses this. You can split the sequence into chunks and only feed the relevant $K, V$ blocks. No masked-out FLOPs.
- **Both are exact:** masking is applied to $S_{ij}$ before the online softmax merge, so the result is bit-identical (modulo FP order) to standard causal attention.

# Quick reference

- Standard attention: $O = \text{softmax}(QK^T)V$, costs $O(N^2)$ HBM and $O(N^2 d)$ FLOPs.
- Flash Attention 2: same FLOPs, $O(N d)$ HBM, $O(N^2)$ compute stays the same. The win is purely memory.
- Inner loop: for each $Q_i$ tile, iterate over all $K_j, V_j$ tiles, maintain $(m_i, \ell_i, O_i)$, apply online softmax merge.
- Typical shapes: $Q, K, V:\ (N, d)$, $Q_i:\ (B_r, d)$, $K_j, V_j:\ (B_c, d)$, $S_{ij}, P_{ij}:\ (B_r, B_c)$, $O_i:\ (B_r, d)$, $m_i, \ell_i:\ (B_r,)$.
- Backward pass: re-tile and recompute $S, P$ on the fly. No $N \times N$ saved.

## Notes

- "FA2" sometimes refers to the original 2022 Flash Attention paper and sometimes to the 2023 revision. The 2023 revision is the one with the work-partitioning fix and is what people mean in practice. Both are exact, both use online softmax, both are IO-aware.
- The output $O$ and the running stats $(m_i, \ell_i)$ must all live in SRAM simultaneously for the merge to work without an HBM round trip. This is what sets the upper bound on $B_r$ and $B_c$ (along with $d$). The Flash Attention 2 paper gives a SRAM budgeting rule of thumb: $B_r \cdot d + B_c \cdot d + B_r \cdot B_c \lesssim \text{SRAM budget}$.
- For very small $N$ (e.g. $N \le 512$), the $N \times N$ attention matrix fits in SRAM anyway and standard attention is competitive. FA2 wins for long sequences where $N^2$ blows past SRAM.
- $d$ here is per-head dim, not the full embedding. Different heads are computed in parallel, so the SRAM budget is per-head. With $d = 128$ and 64 KB of SRAM per CTA, $B_r = B_c = 128$ is comfortable.
- Drop-in replacement for standard attention: same math, same backward, same hyper-parameters. Just call the kernel.

## Why this is the modern default

- **Memory:** attention's $O(N^2)$ cost is the single biggest bottleneck for long-context LLMs. FA2 turns it into $O(N)$ in HBM, which is what enables $32k$, $128k$, $1M$ context windows.
- **Speed:** typically 2-4x faster than standard attention wall-clock, even at $N = 2048$ where you'd think it wouldn't matter. The crossover is around $N = 512$.
- **No quality loss:** exact attention. Perplexity, downstream accuracy, attention maps, all bit-identical (modulo float reduction order). You don't pay a quality tax for the speedup.
- **Hardware friendly:** the algorithm is just nested matmuls with an in-register softmax epilogue. Maps cleanly onto tensor cores. FA3 (2024) takes this further with warp-specialization and async copies, but FA2 is still the baseline everyone implements.

Read the original paper here: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).
