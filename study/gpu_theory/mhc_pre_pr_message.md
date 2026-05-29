## 📌 Description

Hopper-targeted optimization of the `mhc_pre_big_fuse` kernel. The change is gated
behind `#if __CUDA_ARCH__ == 900`, so **all non-Hopper architectures compile the
original code path unchanged** — only `sm_90` (H100) takes the new path, which is
the only GPU it has been profiled on.

Two micro-architectural changes to the per-token metadata stage (the warp-0 work
that produces `pre_mix`/`post_mix`/`comb_mix`):

1. **Remove a 96-byte local-memory spill.** The mix vector was held in a
   `float y_local[24]` indexed by the runtime `lane`, which ptxas placed in local
   memory (`STACK:96`, confirmed via `cuobjdump --dump-resource-usage`). Each active
   lane now loads only its own 6-element slice into scalar / constant-indexed
   registers → `STACK:0`, `LOCAL:0`.
2. **Overlap the Sinkhorn projection with the reduction.** `pre_mix` (the only
   coefficient the `layer_input` reduction consumes) is published *before* the block
   barrier. The 20-iteration Sinkhorn-Knopp normalization that produces `comb_mix`
   on warp 0 now runs concurrently with the bandwidth-bound reduction on warps 1+,
   instead of serializing in front of it.

Numerics are unchanged (identical ops and ordering); the existing reference tests
pass within tolerance. The portable path is byte-for-byte the original kernel.

```text
mHC pre big fuse, enable l2 cache — flashinfer (original) vs optimized (H100, sm_90a)

[BigFuse]
  H=4096                              H=7168
  M/S    original  optimized  x        M/S    original  optimized  x
  1       7.389     6.532    1.13      1       7.816     6.555    1.19
  4       7.375     6.484    1.14      4       7.822     6.535    1.20
  8       7.408     6.509    1.14      8       7.810     6.536    1.19
  16      7.431     6.553    1.13      16      7.841     6.580    1.19
  32      7.478     6.379    1.17      32      7.890     6.586    1.20
  64      7.539     6.589    1.14      64      7.934     6.604    1.20
  128     7.710     6.678    1.15      128     8.183     6.711    1.22
  256     8.297     6.802    1.22      256     9.007     6.885    1.31
  512     9.439     7.168    1.32      512    11.388     7.476    1.52
  1024   17.792     9.817    1.81      1024   32.584    26.982    1.21
  2048   41.702    30.453    1.37      2048   59.322    53.860    1.10
  4096   75.141    58.096    1.29      4096  112.000    97.912    1.14
  8192  137.513   112.465    1.22      8192  210.722   194.560    1.08

[BigFuse+Prenorm]
  H=4096                              H=7168
  M/S    original  optimized  x        M/S    original  optimized  x
  1       8.420     8.264    1.02      1       9.567     9.393    1.02
  8       8.458     8.278    1.02      8       9.649     9.442    1.02
  64      8.556     8.174    1.05      64      9.523     9.557    1.00
  256     9.389     8.798    1.07      256    11.300    10.108    1.12
  512    11.282    10.215    1.10      512    15.439    12.794    1.21
  1024   17.992    13.977    1.29      1024   47.469    43.933    1.08
  2048   51.884    48.201    1.08      2048   92.036    87.798    1.05
  4096  106.198    90.311    1.18      4096  183.625   174.565    1.05
  8192  207.437   187.721    1.11      8192  359.417   347.747    1.03
```

`mhc_post` is not modified by this PR; measured original-vs-this-branch latency is
within ±1% (run-to-run noise) across the full sweep, confirming no regression.
Sub-µs differences at very small M/S are likewise within noise.

## 🔍 Related Issues

Follow-up perf work on the mHC kernels from #3285 (DeepSeek-V4 issue #3346).

## 🚀 Pull Request Checklist

### ✅ Pre-commit Checks

- [✅] `pre-commit run --files csrc/mhc/mhc_pre_big_fuse.cu` — `clang-format`, tabs and
  CRLF hooks all pass (no reformatting required).

## 🧪 Tests

- [x] Existing reference tests pass unchanged on H100:

```bash
  pytest -q tests/mhc/test_mhc_post.py
  pytest -q tests/mhc/test_mhc_pre_big_fuse.py   # 26 passed
```

```bash
  python benchmarks/bench_mhc_pre_big_fuse.py
  python benchmarks/bench_mhc_post.py
```

## Reviewer Notes

- The optimization is **strictly `__CUDA_ARCH__ == 900`**; `sm_80`/`sm_100`/etc. are
  untouched. If reviewers prefer it enabled on other archs, the guard is a one-line
  change once those targets are profiled.
- Profiled on an H100 via Modal. Nsight Compute hardware counters are unavailable in
  that sandbox (`LibraryNotLoaded`), so the analysis used measured latency,
  derived HBM-bandwidth (algorithmic bytes / latency vs the 3.35 TB/s roofline), and
  `cuobjdump --dump-resource-usage` for the register/spill evidence. On the `pure`
  (BigFuse) path the change lifts large-`N` HBM utilization from ~67–78% to ~88–90%.
- The `prenorm` path still makes two streaming passes over `residual` (sqrsum, then
  the weighted-sum reduction); a single-pass shared-memory fusion is a natural
  follow-up but is left out here to keep the change small and `sm_90`-scoped.
