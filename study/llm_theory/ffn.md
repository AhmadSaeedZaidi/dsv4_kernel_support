# Feed-Forward Networks in Transformers

The FFN is the second sub-layer in each transformer block, applied after the attention sub-layer. It processes each token independently (no communication between tokens) through an expansion–activation–contraction pattern. The basic two-matrix form has evolved through three major stages, described below.

# The Original FFN (Vaswani et al., 2017)

The original Transformer uses a simple two-layer feed-forward network with a ReLU activation:

$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

$$
W_1 \in \mathbb{R}^{d \times d_{ff}}, \quad W_2 \in \mathbb{R}^{d_{ff} \times d}, \quad d_{ff} = 4d
$$

## Components

- $d$: model (embedding) dimension, the input and output size of the FFN.
- $d_{ff}$: inner (hidden) dimension. In the original Transformer, $d_{ff} = 4d$, so the hidden layer is 4 times wider than the input.
- $W_1, b_1$: up-projection weights and bias, expanding from $d$ to $d_{ff}$.
- $W_2, b_2$: down-projection weights and bias, compressing from $d_{ff}$ back to $d$.
- $\max(0, \cdot)$: ReLU activation, applied element-wise.

## Why

- **Bottleneck structure:** the expansion gives the model enough dimensions to represent intermediate features, and the compression forces it to be selective. Each token independently passes through this computation.
- **Position-independence:** unlike attention, the FFN processes each token in isolation. This is the counterpart to attention — attention handles inter-token mixing, the FFN handles per-token transformation.
- **Capacity through depth:** stacking FFN layers across transformer blocks allows hierarchical feature extraction. The $4\times$ expansion is empirically motivated — smaller ratios hurt expressivity, larger ratios add compute without proportional gain.

# GeLU Activation (Hendrycks & Gimpel, 2016; popularized by BERT, 2018)

The same two-matrix skeleton, but ReLU is replaced with a smoother non-linearity:

$$
FFN(x) = \text{GeLU}(xW_1 + b_1)W_2 + b_2
$$

$\text{GeLU}(x) = x \cdot \Phi(x)$ where $\Phi(x)$ is the CDF of the standard normal distribution. In practice this is approximated as:

$$
\text{GeLU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715x^3)\right)\right)
$$

more information in [GeLU breakdown](../../notation.md#activation-functions).

## Why GeLU over ReLU

- **Smoother gradients:** GeLU is smooth everywhere. ReLU has a kink at $0$ where the derivative jumps from $0$ to $1$, and its gradient is exactly $0$ for all $x < 0$ — once a neuron enters the negative region, no gradient flows through it. GeLU's gradient is non-zero everywhere, though it is very small for large negative values.
- **Stochastic gating interpretation:** the original GeLU paper motivates it as the expected value of a stochastic regularizer $x \cdot \mathbf{1}_{x > \epsilon}$ where $\epsilon \sim \mathcal{N}(0, 1)$. This gives a theoretical connection to dropout-like regularization while being deterministic at inference.
- **Empirical improvement:** GeLU consistently beats ReLU at the same model size. It was adopted by BERT, GPT-2, and most transformers between 2018-2020, but has since been superseded by GLU variants in newer models.

# GLU Variants (Shazeer, 2020)

Noam Shazeer's paper "GLU Variants Improve Transformer" introduced gated FFN variants. Instead of one expanded vector passed through a fixed activation, GLU variants use an element-wise product of two projections — one activated, one not:

$$
FFN_{\text{SwiGLU}}(x) = \left(\text{SiLU}(xW_1 + b_1) \odot (xW_3 + b_3)\right)W_2 + b_2
$$

$$
FFN_{\text{GeGLU}}(x) = \left(\text{GeLU}(xW_1 + b_1) \odot (xW_3 + b_3)\right)W_2 + b_2
$$

$$
FFN_{\text{ReGLU}}(x) = \left(\max(0, xW_1 + b_1) \odot (xW_3 + b_3)\right)W_2 + b_2
$$

where $\text{SiLU}(x) = x \cdot \sigma(x)$ (also called Swish) and $\odot$ is element-wise multiplication.

## Components

- Three weight matrices instead of two: $W_1, W_3 \in \mathbb{R}^{d \times d_{ff}}$ (gate and up projections), $W_2 \in \mathbb{R}^{d_{ff} \times d}$ (down projection). The two bias vectors follow the same dimensions.
- $W_1$ is the "gate" projection. Its output is passed through an activation (SiLU for SwiGLU, GeLU for GeGLU, ReLU for ReGLU), which then element-wise multiplies with $W_3$'s output. The gate controls which features of the up-projection are allowed through.
- Because there are three matrices instead of two, $d_{ff}$ is typically shrunk to keep the parameter count equal. For the original $4d$ ReLU FFN:
  $$
  3 \cdot d \cdot d_{ff}^\text{GLU} = 2 \cdot d \cdot 4d \;\Longrightarrow\; d_{ff}^\text{GLU} = \frac{8}{3}d \approx 2.67d
  $$

## Why GLU variants

- **Learned gating:** a fixed activation like ReLU or GeLU applies the same shape to every token. The element-wise gate in GLU variants is *learned* — the network can suppress or amplify individual features based on the input, giving finer control over information flow.
- **Better perplexity at equal parameters:** Shazeer (2020) showed that all GLU variants (SwiGLU, GeGLU, ReGLU) outperform their non-gated counterparts at the same parameter budget, even after accounting for the extra weight matrix by reducing $d_{ff}$.
- **Modern standard:** SwiGLU is used in Llama, Llama 2, Llama 3, Mistral, and DeepSeek. GeGLU is used in PaLM. The gated FFN has entirely replaced the original ReLU FFN in large models.

# Quick reference

| Variant | Activation | Weight Matrices | $d_{ff}$ (typical) | Example Models |
|---------|-----------|-----------------|-------------------|----------------|
| ReLU FFN | $\max(0, x)$ | $W_1, W_2$ | $4d$ | Original Transformer (Vaswani 2017) |
| GeLU FFN | $\text{GeLU}(x)$ | $W_1, W_2$ | $4d$ | BERT, GPT-2, GPT-3 |
| SwiGLU | $\text{SiLU}(xW_1) \odot xW_3$ | $W_1, W_2, W_3$ | $\frac{8}{3}d$ | Llama, Mistral, DeepSeek |
| GeGLU | $\text{GeLU}(xW_1) \odot xW_3$ | $W_1, W_2, W_3$ | $\frac{8}{3}d$ | PaLM |
| ReGLU | $\max(0, xW_1) \odot xW_3$ | $W_1, W_2, W_3$ | $\frac{8}{3}d$ | (experimental) |
