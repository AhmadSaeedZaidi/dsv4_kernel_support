# mHC Kernel Profiling & Optimization Report (PR #3285)

Target: the two mHC CUDA kernels in `flashinfer` PR #3285 — `mhc_pre_big_fuse`
and `mhc_post` — profiled and optimized on an H100 80GB (SXM5, HBM3) via Modal.
Shapes: DeepSeek-V4 **Flash** (`H=4096`) and **Pro** (`H=7168`), `HC=4`, bf16.

Harness: [`profiling/profile_mhc_agent.py`](../../profiling/profile_mhc_agent.py).
Correctness gate: the PR's own `tests/mhc` (reference-comparison, rel-norm < 0.6%) —
**26/26 pass before and after** every change.

---

## 1. Measurement methodology

Nsight Compute hardware counters are **unavailable on Modal**: the GPU sandbox does
not expose the PerfWorks driver interface (`Failed to initialize the profiler:
LibraryNotLoaded`; no `ERR_NVGPUCTRPERM`), and the fix requires host-kernel
`NVreg_RestrictProfilingToAdminUsers=0` + reboot — impossible in managed containers.
Modal's docs concur (use torch.profiler / Nsight Systems). We therefore use a
sandbox-safe, rigorous stack:

1. **Measured latency** (authoritative) — the PR benchmarks (`torch.profiler`
   self-time for pre, `bench_gpu_time` for post), cold-L2 rotation for pre.
2. **Derived HBM-BW SOL%** — *algorithmic-minimum* bytes (each distinct input/output
   counted once) ÷ measured latency, vs **3350 GB/s** peak. Valid only for working
   set ≫ L2 (50 MB); smaller shapes are flagged latency-bound.
3. **Static resource usage** — `cuobjdump --dump-resource-usage` on the JIT `.so`
   (registers/thread, stack spills, smem) — exact, GPU-independent.

Byte models (bf16=2, fp32=4, `HC=4`, `MIX=24`):

```
post(N,H)         = 2*N*H*(1+HC+HC) + 4*N*HC + 4*N*HC*HC      # x+res in, out
pre_pure(N,H)     = 2*N*H*(HC+1) + 4*N*MIX + 4*N + ...        # res in, layer out
pre_prenorm(N,H)  = 2*N*H*(HC+1) + 4*N*MIX + ...   (residual counted ONCE)
```

---

## 2. Math → kernel map

mHC replaces the residual skip with $X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$, split
around the heavy block $F_l$:

- **`mhc_pre_big_fuse`** builds the dynamic params and the compressed lane from
  `dot_mix[24] = [pre(4), post(4), comb(16)]` and the RMS scale `rstd`:
  - `pre_mix` $= A_l = \sigma(y\,\mathrm{rstd}\,s_{pre} + b) + \epsilon$  (4)
  - `post_mix` $= C_l = \sigma(\cdot)\cdot m$  (4)
  - `comb_mix` $= B_l$ = Sinkhorn-Knopp doubly-stochastic $4\times4$ (softmax + 20 row/col norms)
  - `layer_input` $= A_l X_l = \sum_j \mathrm{pre}[j]\cdot\mathrm{residual}[j] \to [N,H]$
  - **1 CTA / token**; warp 0 lanes 0–3 do metadata, warps 1+ do the reduction.
- **`mhc_post`** is the epilogue: `out[new,h] = x[h]·C[new] + Σ_old residual[old,h]·B[old,new]`.
  Pure elementwise, **HBM-bandwidth bound** (~`18·N·H` bytes).

---

## 3. Baseline & bottleneck analysis (verified)

| kernel | N | H | latency | HBM SOL% | REG | spill |
|---|---:|---:|---:|---:|---:|---:|
| `mhc_post` persistent | 4096 | 7168 | 181.7 µs | **86.9%** | 71 | 0 |
| `mhc_post` persistent | 4096 | 4096 | 107.5 µs | **84.0%** | 71 | 0 |
| `mhc_pre` **pure** | 4096 | 7168 | 112.0 µs | **78.4%** | 32 | **STACK:96** |
| `mhc_pre` **pure** | 4096 | 4096 | 74.9 µs | **67.2%** | 32 | **STACK:96** |
| `mhc_pre` **prenorm** | 4096 | 7168 | 181.7 µs | **48.4%** | 32 | **STACK:96** |
| `mhc_pre` **prenorm** | 4096 | 4096 | 105.2 µs | **47.8%** | 32 | **STACK:96** |

Latency-bound regime (fixed per-CTA cost): `mhc_pre` pure N=1 ≈ 7.4–7.8 µs, N=128 ≈ 9.9–11.2 µs.

**Bottlenecks identified:**

1. **Local-memory spill (`STACK:96`)** on all `mhc_pre` variants. `float y_local[24]`
   is indexed by the runtime `lane` (`y_local[2*HC + lane*HC + k]`), so ptxas places
   it in local memory (24×4 B = 96 B). It sits on the metadata critical path that 7+
   warps wait for, and all 4 active lanes redundantly load all 24 values.
2. **Sinkhorn warp-0 serialization.** The 20-iteration Sinkhorn (the $B_l$
   doubly-stochastic projection) runs in 4 lanes of warp 0 while warps 1+ block at
   `__syncthreads()`. But `layer_input` depends only on `pre_mix` ($A_l$), computed
   *before* Sinkhorn — so the bandwidth-bound reduction was needlessly serialized
   behind it. The lower SOL at `H=4096` (67%) vs `H=7168` (78%) confirms the fixed
   metadata cost dominates more at smaller `H`.
3. **prenorm double residual pass (48% SOL).** Two streaming passes over `residual`
   (square-sum, then weighted-sum reduction) ≈ 2× the memory work of `pure`.
4. `mhc_post` at 84–87% — already near HBM roofline; left untouched (control).

---

## 4. Optimization applied — `mhc_pre_big_fuse`

Two changes in [`csrc/mhc/mhc_pre_big_fuse.cu`](../source_code/PR_3285_files/csrc/mhc/mhc_pre_big_fuse.cu),
math-preserving (identical ops/order):

**(a) Spill elimination.** The monolithic `write_token_metadata(y_local[24], …)` is
removed; each active lane now loads only its own 6-element slice into scalar
registers (`y_pre`, `y_post`, `cmv[4]` — constant-indexed → registers). No
dynamically-indexed array → no local memory.

**(b) Sinkhorn / reduction overlap.** Metadata is split into two phases around the
existing barrier so warp-0's long Sinkhorn overlaps the warps-1+ reduction:

```cuda
// Phase 1 (warp 0, lanes 0-3): rstd + pre_mix  — the ONLY thing the reduction needs
pre_mix[lane] = sigmoid(y_pre * rstd * scale_pre + base[lane]) + eps;
__syncthreads();
// Phase 2 (warp 0): post_mix + 20-iter Sinkhorn comb_mix   ── overlaps ──┐
if (meta_lane) { post_mix[...] = ...; /* softmax + Sinkhorn */ }          │
write_layer_input<BS>(residual_token, pre_mix, ...);   // warps 1+  <─────┘ HBM-bound
```

`cmv[4]`, `y_post`, `rstd` are held in registers across `__syncthreads()`.
`post_mix`/`comb_mix` write straight to global and gate nothing, so no second
barrier is needed.

---

## 5. Results (verified, 26/26 tests pass)

**Static:** `STACK:96 → 0` (`LOCAL:0`) on every `mhc_pre` variant; registers 32 → 32
(pure) / 38–40 (prenorm) — no spill, occupancy retained.

**Latency & derived HBM SOL:**

| config | baseline | optimized | speedup | SOL% Δ |
|---|---:|---:|---:|---|
| pure N=4096 H=4096 | 74.9 µs | **57.3 µs** | **1.31×** | 67.2 → **87.8** (+20.6) |
| pure N=4096 H=7168 | 112.0 µs | **97.0 µs** | **1.15×** | 78.4 → **90.5** (+12.1) |
| pure N=128 H=7168 | 11.2 µs | **7.4 µs** | **1.52×** | latency-bound |
| pure N=128 H=4096 | 9.9 µs | **7.1 µs** | **1.40×** | latency-bound |
| pure N=1 H=7168 | 7.8 µs | **6.6 µs** | 1.19× | latency-bound |
| prenorm N=4096 H=4096 | 105.2 µs | **88.9 µs** | 1.18× | 47.8 → 56.6 |
| prenorm N=4096 H=7168 | 181.7 µs | **171.7 µs** | 1.06× | 48.4 → 51.2 |
| `mhc_post` (control) | 181.7 µs | 179.6 µs | ~1.0× | 86.9 (stable) |

The **pure** path — the primary production path (consumes precomputed `sqrsum` from
the fused GEMM) — now reaches **88–90% of HBM roofline**, matching the well-tuned
`mhc_post` epilogue, up from 67–78%. The spill fix dominates the small-H / small-N
wins (shorter metadata critical path); the overlap recovers the rest. The prenorm
path benefited from the shared spill fix but remains memory-pass-bound.

---

## 6. Remaining opportunity (future work)

**`mhc_pre` prenorm single-pass fusion** (~51% SOL). Stage `residual` in shared
memory during the square-sum pass and reuse it for the weighted-sum reduction →
one HBM read instead of two. Caveats: dynamic smem `4·H·2` B (`H=7168 → 56 KB`
needs the `>48 KB` opt-in; very large `H` needs a 2-pass fallback), and occupancy
drops to ~4 CTAs/SM at 56 KB — net win requires measurement. Estimated upside:
prenorm `H=7168` ~172 µs → approaching pure's ~97 µs.

`mhc_post` is already at the bandwidth roofline; no change recommended.
