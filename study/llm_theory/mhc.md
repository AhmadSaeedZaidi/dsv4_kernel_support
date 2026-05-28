# Manifold Constraint Hyper Connection (MHC)

Replaces the standard residual skip connection $x_{l+1} = F(x_l) + x_l$ with a structured "hyper-connection" that mixes and routes multiple embedding lanes.

Motivation: residual skips help gradients flow (since $x_{l+1}' = F'(x_l) + I$). MHC generalizes this by (1) projecting multiple lanes into a smaller compute lane, (2) applying a heavy block (MoE/Attention) there, and (3) redistributing the result back to the full lane dimension.

## Standard hyper-connection

The HC layer is written compactly as

$$
X_{l+1} = B_l X_l + C_l\,F_l\big(A_l X_l\big)
$$

where the terms are described below.

## Components

- `X_l`, `X_{l+1}`: input and output of layer $l$. Typically `X_l` has shape $(n_{hc}, d)$.
- $n_{hc}$: the number of hyper-connection lanes.
- $d$: the embedding width.
- $A_l$: projection that squeezes the $n_{hc}$ lanes into a single lane. Shape: $A_l:\ (1, n_{hc})$, so $A_l X_l$ has shape $(1, d)$. Effect: compress many lanes into 1.
- $F_l(\cdot)$: the expensive compute block (e.g. MoE or multi-head attention) that operates on the compressed $(1, d)$ lane and returns $(1, d)$.
- $C_l$: expansion (or coloring) matrix that maps the $(1, d)$ output of $F_l$ back to $(n_{hc}, d)$. Shape: $C_l:\ (n_{hc}, 1)$, so $C_l\,F_l(A_l X_l)$ has shape $(n_{hc}, d)$.
- $B_l$: bypass mixing matrix applied directly to the original $n_{hc}$ lanes. Shape: $B_l:\ (n_{hc}, n_{hc})$; it mixes the bypassed input before adding to the expanded path.
## Why
- we make a massive dimension d = 7168, each value in this vector is a feature of the token
- we use multi-head attention, each head in the 7168 embedding tries to learn a single group of features (grammer, semantics, context), and they fuse at the end using some kind of projection. After a lot of $+x$ operations, the output stiffens (100 layers of operations meld into 1 brown turd)
- we use multiple channels (eg. 4), each with a full 7168 embedding. Each channel is a global stream that learns a different feature. Before this, the single vector was overburdenned, and prone to losing some knowledge as it passed through attention layers.

## Notes

- The ordering of linear ops shown above is convenient for exposition; implementations often fuse projections or use learned channel-wise gates.
- Using a single compressed lane for the heavy block reduces compute and memory compared with applying the block to all lanes.
- Replace or tune the shapes ($n_{hc}$ and $d$) to match your model's lane count and embedding dimension.

## Quick reference

- Equation: $X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$
- Typical shapes: $X_l:\ (n_{hc}, d)$, $A_l:\ (1, n_{hc})$, $F_l:\ (1, d)\to(1, d)$, $C_l:\ (n_{hc}, 1)$, $B_l:\ (n_{hc}, n_{hc})$

## MHC
$$ 
B \in M := {M \in R^{n \times n}} | \sum_{row = 1}^{n} M_{row} = 1, \sum_{col = 1}^{n} M_{col} = 1, M_{element} \ge 1
$$