# Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)
Deepseek introduces a hybrid attention architecture, that utilizes both CSA and HCA. The workings of these attention mechanisms are detailed below.
## Compressed Sparse Attention (CSA)
$$
H \in R^{n \times d}
$$
H is a hidden state, entering the attention block. with $n$ tokens/seq length and $d$ hidden size.

$$
C^a,C^b \in R^{n \times c} \space | \space C^a = H \cdot W^{aKV}, C^b = H \cdot W^{bKV}
$$

$$
Z^a,Z^b \in R^{n \times c} \space | \space Z^a = H \cdot W^{aZ}, Z^b = H \cdot W^{bZ}
$$

$C^a$ and $C^b$ are KV entries, each C represents both K and V. The reason we have a and b, will become clearer in the next section.

$Z^a$ and $Z^b$ represent "token importance" scores, basically how much weight each token should have in the attention calculation.

$$
W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ} \in R^{d \times c}
$$

where $W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ}$ are learnable weight matrices, and $c$ is the head dimension.

now, the matrix $C^a$ and $C^b$ are split into contiguous chunks of size $m = 4$ for compressed sparse attention in the paper. So take for example the chunks $(i-1)$, $i$, $(i+1)$, we have:
$$
S^a \in R^{n \times c} \space | \space S_{m\cdot i\rightarrow m\cdot(i+1)-1}^a = S^{a'}
$$

$$
S^b \in R^{n \times c} \space |\space S_{m\cdot (i-1)\rightarrow m \cdot i-1}^b = S^{b'}
$$
$$
[S^{a'};S^{b'}] = Softmax_{row}([Z_{m\cdot i\rightarrow m\cdot(i+1)-1}^a + B^a; Z_{m\cdot (i-1)\rightarrow m \cdot i-1}^b + B^b])
$$

This is a bit of confusing notation in the paper. $S^{a'}$ consists of the rows of block $i$ of $S^a$. $S^{b'}$ consists of the rows of block $(i-1)$ of $S^b$.

To compute them, we first take the m rows corresponding to block $i$ of $Z^a$ and block $(i-1)$ of $Z^b$ respectively. We then add bias term $B^a$ and $B^b$ to them respectively, and then we concatenate them together, creating a matrix with $2\cdot m$ rows. Then we take row-wise softmax, along each row of $2 \cdot m$ rows, and we get $S^{a'}$ and $S^{b'}$ concatenated together $[S^{a'};S^{b'}]$. We can then split them back into $S^{a'}$ and $S^{b'}$. We do this $\frac{n}{m}$ times, for each block of rows in $Z^a$ and $Z^b$.

This is why we need $aKV$ and $bKV$, because we need to compute them in pairs. Where $bKV$ acts as a history buffer for the previous block of tokens, and $aKV$ is the current block. 

Finally, we compute our compressed KV output as:

$$
C_i^{comp} = \Sigma _{j=m \cdot i}^{m \cdot (i+1)-1} S_j^a \odot C_j^a + \Sigma _{j=m \cdot (i-1)}^{m \cdot i-1} S_j^b \odot C_j^b
$$ 

where $\odot$ is element-wise multiplication. We do this for each block $i$, and we concatenate the results together to get our final compressed KV output. As noted above, this is a copression of $1/m$ times.

furthermore:

$$
Z_{-m\rightarrow -1}^b = -\infty \qquad C_{-m\rightarrow -1}^b = 0
$$

we need to pad an extra block of $m$ rows to the beginning of $Z^b$ and $C^b$, with $-\infty$ for $Z^b$ and 0 for $C^b$.
## Lightning Indexer for Sparse Selection
