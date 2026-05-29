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

## Set Notation
- $M := \{M \in R^{n \times n} | \text{conditions}\}$ means we are defining a set M that contains all matrices of shape $n \times n$ that satisfy the specified conditions. For example, $M := \{M \in R^{n \times n} | M_{element} \ge 0\}$ would be the set of all $n \times n$ matrices with non-negative elements.

## Concatenation of Matrices
- When I write $W_l^{fused} = [W_l^{pre}; W_l^{res}; W_l^{post}]$, I mean that we are concatenating the three matrices $W_l^{pre}$, $W_l^{res}$, and $W_l^{post}$ along the appropriate dimension to create a single fused weight matrix. The exact way they are concatenated depends on their shapes, but the idea is to combine them into one larger matrix that can be used in a single matrix multiplication operation for efficiency.

- for a simple example if I have vector $a = [1, 2]$ and $b = [3, 4]$, then concatenating them as $c = [a; b]$ would give us $c = [1, 2, 3, 4]$. In the case of matrices, the concatenation would be done along a specific axis (e.g., rows or columns) depending on the shapes of the matrices being concatenated.