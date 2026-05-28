# My efforts at contributing to flashinfer's [deepseek v4 project](https://github.com/flashinfer-ai/flashinfer/issues/3346)

## 5/27: found source code [mhc_pre.py's pytorch](study/source_code/mhc_pre.py)
so the main functions (written in `tilelang`) are:
- [_compute_num_split_for_mhc_pre](study/source_code/mhc_pre.py#_compute_num_split_for_mhc_pre) 
- [mhc_pre_gemm_sqrsum_tilelang](study/source_code/mhc_pre.py#mhc_pre_gemm_sqrsum_tilelang)
- [mhc_pre_gemm_sqrsum_splitk_kernel](study/source_code/mhc_pre.py#mhc_pre_gemm_sqrsum_splitk_kernel)
- [mhc_pre_big_fuse_with_norm_tilelang](study/source_code/mhc_pre.py#mhc_pre_big_fuse_with_norm_tilelang)
- [mhc_pre_big_fuse_tilelang](study/source_code/mhc_pre.py#mhc_pre_big_fuse_tilelang)

dispatched by:
- [mhc_pre](study/source_code/mhc_pre.py#mhc_pre)

## 5/28: studying how mhc works
- [mhc notes](study/llm_theory/mhc.md)

- explained how standard hyper channel residuals work [standard hyper-connection](study/llm_theory/mhc.md#standard-hyper-connection)
- explained the manifold constraints applies to the residual mapping in new deepseek v4 mHC [manifold constraint](study/llm_theory/mhc.md#manifold-constrained-residual-mapping)
- explained the dynamic mappings for the parameters of the mHC [dynamic parameterization](study/llm_theory/mhc.md#dynamic-parameterization)
- explained how the manifold constraints are implemented in the dynamic parameterization [parameter constraints](study/llm_theory/mhc.md#parameter-constraints-and-regularization)

## 5/29: studying how the GPU kernels work
- [GPU optimization notes](study/gpu_theory/hc_pre.md)