# Mathematical Notation Guide
this guide is for students who have strong logical and programming skills but are new to mathematical notation. It explains the symbols and conventions used in the DeepSeek V4 notes, especially in the MHC design and GPU kernel discussions.
## Summation (Sigma) Notation

$$
\Sigma(\cdot)_{i=1}^n \text{ is the summation operator} \qquad
$$
generally means "sum up the following expression for i from 1 to n".
for example 
$$
\Sigma_{i=1}^n i = 1 + 2 + 3 + ... + n
$$

I've used it in a weird way, like $\Sigma_{row=1}^n M_{row}$ to mean "sum up all the rows of the matrix M". This is just a notational choice to indicate summing across a specific dimension. This would create a vector where each element is the sum of the corresponding row in M. 

## Matrix Dimensions and Shapes
- $A \in R^{m \times n}$ means A is a matrix with m rows and n columns. The first number is the number of rows, and the second is the number of columns. Formally it means A is an element of the set of all real-valued matrices with m rows and n columns, or that A belongs to a real space with $m \times n$ dimensions.
- When we say $A X$, it implies matrix multiplication. If $A$ is $m \times n$ and $X$ is $n \times p$, then the result will be $m \times p$.

- if you practice matrix multiplication on a pen and paper, you will notice that for 
$$
A \in R^{m \times n}, X \in R^{n \times p} \implies A X \in R^{m \times p}
$$
because the inner dimensions (the n's) must match for multiplication to be valid, and the resulting matrix takes the outer dimensions (m and p).

- In the context of tensors, we often use the term "shape" to describe the dimensions of the tensor. For example, if we say $X$ has shape $(n_{hc}, d)$, it means $X$ is a matrix with $n_{hc}$ rows and $d$ columns.

## Element-wise Operations
- $A \odot B$ denotes element-wise multiplication (Hadamard product) between two matrices A and B of the same shape. Each element in the resulting matrix is the product of the corresponding elements in A and B.
- $\sigma(\cdot)$ typically denotes an element-wise activation function, such as the sigmoid function. When you see $\sigma(\~A_l)$, it means we are applying the activation function to each element of the matrix $\~A_l$.
- When I write $A_{element} \ge 0$, it means that every element in the matrix A must be greater than or equal to zero. This is a common constraint for certain types of matrices, such as those used in attention mechanisms or gating functions.

## Matrix Multiplication vs. Element-wise Operations
- When you see $A X$, it means matrix multiplication, which involves summing over the products of rows of A and columns of X.
- When you see $A \odot X$, it means element-wise multiplication, where you multiply each corresponding element of A and X together without any summation.

## Element-wise Division
- $A \oslash B$ denotes element-wise division between two matrices A and B of the same shape. Each element in the resulting matrix is the quotient of the corresponding elements in A and B. For example, if $A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$ and $B = \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix}$, then 
$$
A \oslash B = 
\begin{bmatrix} 
a_{11} / b_{11} & a_{12} / b_{12} \\ 
a_{21}/b_{21} & a_{22}/b_{22}
\end{bmatrix}
$$

### Mismatched Dimensions
- If you see an expression like $A \oslash X$ where the dimensions of A and X do not match, it typically means that A is being broadcasted to match the shape of X before performing element-wise division. 
- Broadcasting is a technique that allows operations to be performed on arrays of different shapes by automatically expanding the smaller array along the necessary dimensions. This is often used in computing to simplify notation.
- For example, if $A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$ and $B = \begin{bmatrix} b_{11} \\ b_{21} \end{bmatrix}$, then 
$$
A \oslash B = A \oslash [B;B] =
\begin{bmatrix} 
a_{11} / b_{11} & a_{12} / b_{11} \\ 
a_{21}/b_{21} & a_{22}/b_{21}
\end{bmatrix}
$$

Here I am using [concatonation](#concatenation-of-matrices) defined below, to match the shape of A by repeating B along the appropriate dimension.

## Set Notation
- $M := \{M \in R^{n \times n} | \text{conditions}\}$ means we are defining a set M that contains all matrices of shape $n \times n$ that satisfy the specified conditions. For example, $M := \{M \in R^{n \times n} | M_{element} \ge 0\}$ would be the set of all $n \times n$ matrices with non-negative elements.

## Piecewise (Case) Notation

- A piecewise function assigns different expressions depending on a condition. The notation is:
  $$
  f(x) = \begin{cases}
  \text{expression}_1 & \text{if condition}_1 \\
  \text{expression}_2 & \text{if condition}_2
  \end{cases}
  $$
- For example, the causal attention mask is:
  $$
  M_{ij} = \begin{cases}
  0 & j \le i \\
  -\infty & j > i
  \end{cases}
  $$
  meaning "put 0 if $j \le i$, otherwise put $-\infty$".
- And the KeepTopK function used in the MoE router (see [moe.md](study/llm_theory/moe.md)):
  $$
  \text{KeepTopK}(v, k)_i = \begin{cases}
  v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v \\
  -\infty & \text{otherwise}
  \end{cases}
  $$
  meaning "keep the value if it's among the $k$ largest, otherwise discard it by setting it to $-\infty$". After softmax, the $-\infty$ entries become exactly $0$.

## Activation Functions

An activation function is a non-linear transformation applied element-wise to a vector or matrix. The ones used in these notes:

- **ReLU** (Rectified Linear Unit): $\text{ReLU}(x) = \max(0, x)$. Returns $x$ if positive, $0$ otherwise. Simple and cheap, but has zero gradient for $x < 0$. Used in the original Transformer FFN.

- **GeLU** (Gaussian Error Linear Unit): $\text{GeLU}(x) = x \cdot \Phi(x)$ where $\Phi(x)$ is the CDF of the [standard normal distribution](#derivation-of-phi-x-in-gelu).
  
  We use a numerical approximation of the function:
  $$
  \text{GeLU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)
  $$

  GeLU is smooth everywhere and has non-zero gradient for negative inputs. It is not defined in these notes beyond this section, but it is used in BERT/GPT-2 and the ReGLU, and GeGLU variants of the FFN (see [ffn.md](study/llm_theory/ffn.md)).

- **Sigmoid**: $\sigma(x)$. Maps real numbers to the range (0, 1). Used in most deep neural networks for example in the gating network of the MoE design (see [moe.md](study/llm_theory/moe.md)).  The sigmoid function is defined as:
$$
\sigma(x) = \frac{1}{1 + e^{-x}} \in [0, 1]
$$

- **SiLU** (Sigmoid Linear Unit, also called **Swish**): $\text{SiLU}(x) = x \cdot \sigma(x)$ where $\sigma(x)$ is the sigmoid function. Like GeLU, it is smooth and non-monotonic. Used in the **SwiGLU** variant of the FFN.

- **Tanh**: A smooth, S-shaped function that maps real numbers to the range [-1, 1]. Used in the GeLU approximation. It is basically the function that maps the behaviour of $-i\cdot \tan(i x)$ 
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in [-1, 1]$$

- **Softplus**: $\text{Softplus}(x) = \log(1 + e^x)$. A smooth, strictly positive approximation to ReLU. Used in the MoE gating network to ensure noise magnitudes are positive (see [moe.md](study/llm_theory/moe.md)).

## Softmax Function

- $\text{Softmax}(x)$ converts a vector of arbitrary real numbers into a probability distribution where all entries sum to 1. For a vector $x \in \mathbb{R}^n$:

  $$
  \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
  $$
- The exponential makes all values positive, and the denominator ensures they sum to 1.
- $\text{Softmax}_{row}(M)$ means applying softmax to each row of a matrix $M$ independently, so each row sums to 1. This is how we get attention weights from scores.

## Statistical Notation

- $\mathbb{E}[X]$: expected value (mean) of a random variable $X$. For a finite set of values $X = \{x_1, \dots, x_n\}$, the mean is $\mathbb{E}[X] = \frac{1}{n}\sum_{i=1}^n x_i$.
- $\text{Std}[X]$: standard deviation of $X$, measuring how spread out the values are. $\text{Std}[X] = \sqrt{\text{Var}[X]}$.
- $\text{CV}(X)$: coefficient of variation, $\text{CV}(X) = \frac{\text{Std}[X]}{\mathbb{E}[X]}$. This normalizes the spread by the mean, giving a unitless measure of relative dispersion used in the MoE importance loss.
- $\mathcal{N}(0, 1)$: the standard normal distribution (bell curve) with mean $0$ and standard deviation $1$. Samples are real numbers, mostly between $-3$ and $3$. Written $\mathcal{N}(0, 1)$ in short, it's short for "normal distribution with mean 0 and variance 1". Used in the MoE gating network to add random noise for exploration.

## Concatenation of Matrices
- When I write $W_l^{fused} = [W_l^{pre}; W_l^{res}; W_l^{post}]$, I mean that we are concatenating the three matrices $W_l^{pre}$, $W_l^{res}$, and $W_l^{post}$ along the appropriate dimension to create a single fused weight matrix. The exact way they are concatenated depends on their shapes, but the idea is to combine them into one larger matrix that can be used in a single matrix multiplication operation for efficiency.

- for a simple example if I have vector $a = [1, 2]$ and $b = [3, 4]$, then concatenating them as $c = [a; b]$ would give us $c = [1, 2, 3, 4]$. In the case of matrices, the concatenation would be done along a specific axis (e.g., rows or columns) depending on the shapes of the matrices being concatenated.

## Derivation of $\Phi(x)$ in GeLU
$\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution. The standard normal distrbution is a bell curve defined by:
$$
\Large
f(x) = \frac{1}{\sqrt{2\pi}} e^{\Large -\tfrac{t^2}{2}} 
$$

The CDF $\Phi(x)$ is the integral of the PDF from $-\infty$ to $x$:
$$
\Large \Phi(x) = \int_{-\infty}^x f(t) dt = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{\Large -\tfrac{t^2}{2}} dt
$$

This integral does not have a closed-form solution in terms of elementary functions, which is why we use numerical approximations for $\Phi(x)$ when implementing GeLU. The approximation I provided earlier is a commonly used one that balances accuracy and computational efficiency.

to derive it, we use the error function $\text{erf}(x)$, which is related to the normal distribution:
$$
\Large \text{erf}(x) =  \int_0^x \frac{2}{\sqrt{\pi}}  e^{-t^2}dt
$$
as you might notice, this is similar to the integral for $\Phi(x)$, except it starts from 0 and has a different scaling factor. We can express $\Phi(x)$ in terms of $\text{erf}(x)$:

first we split the integral for $\Phi(x)$ into two parts:
$$
\Phi(x) = \int_{-\infty}^0 f(t) dt + \int_0^x f(t) dt
$$
the first part evaluates to 0.5. This is because the total probability of the normal distribution is 1, and the distribution is symmetric around 0, so from $-\infty$ to 0 is half of the total area under the curve.
$$
\int_{-\infty}^0 f(t) dt = 0.5
$$

we use a u substitution to evaluate the second part:
let 
$$
u = \frac{t}{\sqrt{2}}
$$
then 
$$
t = u \sqrt{2} \qquad dt = \sqrt{2} du
$$
Substituting this gives
$$
\int_0^x f(t) dt = \int_0^{\frac{x}{\sqrt{2}}} \frac{1}{\sqrt{2\pi}} e^{\Large -\tfrac{(u \sqrt{2})^2}{2}} \sqrt{2} du = \int_0^{\frac{x}{\sqrt{2}}} \frac{1}{\sqrt{\pi}} e^{-u^2} du = \frac{1}{2} \text{erf}\left(\frac{x}{\sqrt{2}}\right)
$$

substituting this back into the expression for $\Phi(x)$ gives us:
$$
\Phi(x) = 0.5 + \frac{1}{2} \text{erf}\left(\frac{x}{\sqrt{2}}\right) = \frac{1}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$ 

To get a practical approximation, we express $\text{erf}(x/\sqrt{2})$ in terms of $\tanh$, which is fast to compute on hardware. We fit a low-degree polynomial inside $\tanh$ so the Taylor expansions match up to a chosen order.

The Taylor series of $\text{erf}(x/\sqrt{2})$ around $x = 0$ is:

$$
\text{erf}\!\left(\frac{x}{\sqrt{2}}\right) = \sqrt{\frac{2}{\pi}}\left(x - \frac{x^3}{6} + \frac{x^5}{40} - \frac{x^7}{336} + \cdots\right)
$$

The Taylor series of $\tanh$ around $0$ is:

$$
\tanh(y) = y - \frac{y^3}{3} + \frac{2y^5}{15} - \frac{17y^7}{315} + \cdots
$$

Now let $y = \sqrt{\frac{2}{\pi}}(x + a x^3)$ and expand $\tanh(y)$. Work term by term:

- The $x$ term: $\sqrt{\frac{2}{\pi}}$, which already matches erf's leading coefficient.
- The $x^3$ term comes from two parts:
  - From $y$ itself: $\sqrt{\frac{2}{\pi}} \cdot a$
  - From $-\frac{y^3}{3}$: $-\frac{1}{3}\left(\sqrt{\frac{2}{\pi}}\right)^3 = -\frac{1}{3}\left(\frac{2}{\pi}\right)^{3/2}$

    Together: $\sqrt{\frac{2}{\pi}} \, a - \frac{1}{3}\left(\frac{2}{\pi}\right)^{3/2}$

We want this to match erf's $x^3$ coefficient, which is $-\frac{1}{6}\sqrt{\frac{2}{\pi}}$:

$$
\sqrt{\frac{2}{\pi}} \, a - \frac{1}{3}\left(\frac{2}{\pi}\right)^{3/2} = -\frac{1}{6}\sqrt{\frac{2}{\pi}}
$$

Divide both sides by $\sqrt{2/\pi}$:

$$
a - \frac{1}{3} \cdot \frac{2}{\pi} = -\frac{1}{6}
$$

Solve for $a$:

$$
a = \frac{2}{3\pi} - \frac{1}{6} = \frac{4 - \pi}{6\pi} \approx 0.04554
$$

This is very close to the $0.044715$ used in the actual approximation, but not exact. The small difference comes from the fact that matching the $x^5$ term would require a small $x^5$ correction inside $\tanh$. When we instead stick with just $x + a x^3$ and re-tune $a$ by minimizing the maximum absolute error between $\tanh(\sqrt{2/\pi}(x + a x^3))$ and $\text{erf}(x/\sqrt{2})$ over a range (say $x \in [-2, 2]$), the optimal value shifts slightly from the pure Taylor result to $0.044715$. This numerically-refined value gives a better overall fit than the Taylor-only value because it balances errors across higher-order terms instead of exactly matching only the $x^3$ term.

The final approximation for GeLU is therefore:

$$
\text{GeLU}(x) \approx x \cdot \frac{1}{2}\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right)
$$

with a maximum error of about $0.001$ compared to the true $x \cdot \Phi(x)$ over $x \in [-4, 4]$ — more than accurate enough for neural network training.

This analysis is verified by [`study/llm_theory/verify_gelu_approx.py`](study/llm_theory/verify_gelu_approx.py), which brute-force searches for the minimax-optimal $a$ over a dense grid. The script confirms:
- $a_\text{taylor} \approx 0.04553992$, max error $6.27 \times 10^{-4}$
- $a_\text{paper} = 0.04471500$, max error $3.58 \times 10^{-4}$
- The paper's value is already at the minimax optimum (the brute-force minimax converges to $0.044715$).