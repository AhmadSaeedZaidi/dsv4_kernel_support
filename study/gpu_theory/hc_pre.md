# The Hyper Connection Preprocessing GPU Kernel Study
This document summarizes my study of the GPU kernels used for the hyper-connection (mHC) preprocessing in FlashInfer's DeepSeek V4. It covers my exploration of the reference implementation, the GPU optimization techniques involved (like split-K and tiling), and notes on how these kernels work to support the mHC design.
[diagram](study/media/hc_diagram.png) is a useful visual reference for understanding the data flow and operations in the mHC preprocessing.


## dispatcher mhc_pre()
[mhc_pre function](../source_code/mhc_pre.py#), is the main entry point for the mHC preprocessing. It takes in the following tensors:
```
residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
```
