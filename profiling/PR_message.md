<!-- .github/pull_request_template.md -->

## 📌 Description

  This PR adds mHC (multi-head hyper-connection) CUDA kernels and Python APIs to FlashInfer, targeting DeepSeek-V4-style post/pre mapping workloads.

  Main changes:

  - Add `flashinfer.mhc_post` for mHC post mapping:
    - `out[..., new_hc, h] = x[..., h] * post_mix[..., new_hc] + sum_old residual[..., old_hc, h] * comb_mix[..., old_hc, new_hc]`
    - Supports HC=4 and bf16 residual/input tensors.
  - Add mHC pre big-fuse finalize APIs:
    - `flashinfer.mhc_pre_big_fuse(...)`: consumes precomputed `dot_mix` and `sqrsum`, supports split reduction for `num_splits in {1, 2, 4, 8, 16}`.
    - `flashinfer.mhc_pre_big_fuse_with_prenorm(...)`: consumes precomputed `dot_mix` and computes RMS sqrsum from `residual` inside the kernel.
  - Add JIT wiring for the mHC kernels.
  - Add correctness tests for post mapping and pre big-fuse paths.
  - Add microbenchmarks for mHC post mapping and pre big-fuse.

  The post mapping kernel is optimized for the common HC=4 shapes and includes special paths for hidden sizes such as 4096 and 7168.

  ```text
  mHC post mapping cold-L2, flashinfer vs TensorRT-LLM

  H=4096
  N      flashinfer   TensorRT-LLM
  1      7.168        9.232
  32     7.136        9.312
  64     7.200        11.264
  128    9.184        11.264
  256    11.232       11.328
  512    14.176       13.376
  1024   21.440       21.504
  2048   33.760       33.728
  4096   56.288       56.352
  8192   101.376      101.424

  H=7168
  N      flashinfer   TensorRT-LLM
  1      9.248        11.296
  32     11.232       13.312
  64     11.232       13.344
  128    11.232       13.344
  256    13.280       15.360
  512    21.408       19.488
  1024   29.728       31.776
  2048   50.144       50.208
  4096   90.176       91.136
  8192   168.896      168.960
```

mHC pre map big fuse, enable l2 cache (residual input will be hold in l2 cache)

``` text

  H=4096
  M/S    BigFuse       BigFuse+Prenorm
  1      3.7           4.6
  4      3.7           4.7
  8      3.7           4.7
  16     4.6           5.2
  32     4.8           5.4
  64     4.9           5.5
  128    5.0           5.6
  256    5.2           6.0
  512    5.9           7.1
  1024   8.4           9.3
  2048   15.2          17.5
  4096   33.2          39.3
  8192   68.2          73.6

  H=7168
  M/S    BigFuse       BigFuse+Prenorm
  1      4.3           5.6
  4      4.2           5.9
  8      4.3           6.2
  16     5.7           6.7
  32     5.8           7.1
  64     5.9           7.1
  128    6.1           7.2
  256    6.3           7.5
  512    7.3           9.6
  1024   11.9          13.7
  2048   26.7          30.8
  4096   52.3          62.7
  8192   104.8         119.9
```

## 🔍 Related Issues

<!-- Link any related issues here -->

## 🚀 Pull Request Checklist

Thank you for contributing to FlashInfer! Before we review your pull request, please make sure the following items are complete.

### ✅ Pre-commit Checks

- [✅] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [✅] I have installed the hooks with `pre-commit install`.
- [✅] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

> If you are unsure about how to set up `pre-commit`, see [the pre-commit documentation](https://pre-commit.com/).

## 🧪 Tests

- [x] Tests have been added or updated as needed.
- [x] All tests are passing (`unittest`, etc.).

``` bash
  pytest -q tests/mhc/test_mhc_post.py
  pytest -q tests/mhc/test_mhc_pre_big_fuse.py
```

  Relevant benchmarks:

``` bash
  python benchmarks/bench_mhc_post.py
  python benchmarks/bench_mhc_pre_big_fuse.py
```

## Reviewer Notes

<!-- Optional: anything you'd like reviewers to focus on, concerns, etc. -->


<!-- This is an auto-generated comment: release notes by coderabbit.ai -->
## Summary by CodeRabbit

* **New Features**
  * Added CUDA‑accelerated MHC post/pre processing ops (including prenorm variant), package‑level exposure, a JIT loader, configurable split handling, and new trace templates.

* **Tests**
  * Added CUDA‑only tests that validate numerical accuracy and output shapes against reference implementations.

* **Chores**
  * Added GPU benchmarking tools for latency and cold‑L2 profiling with configurable grids and tabular reporting.

<!-- review_stack_entry_start -->

[![Review Change Stack](https://storage.googleapis.com/coderabbit_public_assets/review-stack-in-coderabbit-ui.svg)](https://app.coderabbit.ai/change-stack/flashinfer-ai/flashinfer/pull/3285?utm_source=github_walkthrough&utm_medium=github&utm_campaign=change_stack)

<!-- review_stack_entry_end -->
<!-- end of auto-generated comment: release notes by coderabbit.ai -->