# Manifold Constraint Hyper Connection (MHC)

Replaces the standard residual skip connection $x_{l+1} = F(x_l) + x_l$ with a structured "hyper-connection" that mixes and routes multiple embedding lanes.

Motivation: residual skips help gradients flow (since $x_{l+1}' = F'(x_l) + I$). MHC generalizes this by (1) projecting multiple lanes into a smaller compute lane, (2) applying a heavy block (MoE/Attention) there, and (3) redistributing the result back to the full lane dimension.

## Standard hyper-connection

The MHC layer is written compactly as

$$
X_{l+1} = B_l X_l + C_l\,F_l\big(A_l X_l\big)
$$

where the terms are described below.

## Components

- `X_l`, `X_{l+1}`: input and output of layer $l$. Typically `X_l` has dimension $4d$ (i.e. 4 lanes of embeddings).
- $A_l$: projection that squeezes the $4d$ input into a single $d$-sized lane. Shapes (applied as $A_l X_l$):  $A_l:\ (d, 4d)$, $X_l:\ (4d,1)$ → result $(d,1)$. Effect: compress 4 lanes into 1.
- $F_l(\cdot)$: the expensive compute block (e.g. MoE or multi-head attention) that operates on the compressed $(d,1)$ lane and returns $(d,1)$.
- $C_l$: expansion (or coloring) matrix that maps the $(d,1)$ output of $F_l$ back to $(4d,1)$. Shape: $C_l:\ (4d,d)$, so $C_l\,F_l(A_l X_l)$ has shape $(4d,1)$.
- $B_l$: bypass mixing matrix applied directly to the original $4d$ lanes. Shape: $B_l:\ (4d,4d)$; it mixes the bypassed input before adding to the expanded path.
## Why
- we make a massive dimension d = 7168, each value in this vector is a feature of the token
- we use multi-head attention, each head in the 7168 embedding tries to learn a single group of features (grammer, semantics, context), and they fuse at the end using some kind of projection. After a lot of $+x$ operations, the output stiffens (100 layers of operations meld into 1 brown turd)
- we use multiple channels (eg. 4), each with a full 7168 embedding. Each channel is a global stream that learns a different feature. Before this, the single vector was overburdenned, and prone to losing some knowledge as it passed through attention layers.

## Notes

- The ordering of linear ops shown above is convenient for exposition; implementations often fuse projections or use learned channel-wise gates.
- Using a single compressed lane for the heavy block reduces compute and memory compared with applying the block to all lanes.
- Replace or tune the shapes ($4d$ and $d$) to match your model's lane count and embedding dimension.

## Quick reference

- Equation: $X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$
- Typical shapes: $X_l:\ (4d,1)$, $A_l:\ (d,4d)$, $F_l:\ (d,1)\to(d,1)$, $C_l:\ (4d,d)$, $B_l:\ (4d,4d)$