### 1\.Occupancy Router: `_compute_num_split_for_mhc_pre`

This function is solving a classic GPU occupancy problem known as **Split-K**.

In a standard GEMM (Matrix Multiplication), the workload is parallelized across the `M` (tokens) and `N` (output channels) dimensions.

-   If you have a long prompt (e.g., `num_tokens = 4096`), `M` is huge. The GPU has plenty of work to distribute across its SMs (Streaming Multiprocessors).
-   **The Problem:** During generation (decode phase) or with very short prompts, `num_tokens` is tiny (e.g., 1 to 128). `M` is too small to fill the H100's 132 SMs. Most of the GPU will sit idle.
-   **The Solution (Split-K):** Instead of just splitting work by `M` and `N`, we also split the dot product itself along the `K` dimension (the `hc_hidden_size`). Multiple thread blocks compute partial dot products for the exact same output tile, and we sum them up at the end.

**What the code is doing:** It looks at `grid_size` (how many blocks we get just from splitting `M`). If that number is significantly lower than `n_sms` (total SMs on the GPU), it calculates a multiplier (`num_block_k // 4`) to slice the `K` dimension, artificially creating more blocks to keep the GPU fed.