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

- explained how standard hyper channel residuals work (deepseek v3)
- explained the manifold constraints applies to the residual mapping in new deepseek v4 mHC [manifold constraint](study/llm_theory/mhc.md#manifold-constrained-residual-mapping)