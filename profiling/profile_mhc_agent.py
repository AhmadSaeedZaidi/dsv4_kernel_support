"""
profile_mhc_agent.py — Remote H100 profiler for FlashInfer PR #3285 mHC kernels.

The local host has no CUDA GPU/toolchain, so ALL compilation, execution, and
measurement happens inside a remote Modal H100 container.

Metric methodology (Modal-viable)
---------------------------------
Modal runs GPU containers in a sandbox that does NOT expose the GPU
performance-counter interface Nsight Compute (`ncu`) requires (confirmed:
`Failed to initialize the profiler: LibraryNotLoaded`, and Modal's own docs
direct users to torch.profiler / Nsight Systems). So instead of ncu SOL pages
we use a rigorous, sandbox-safe stack:

  1. MEASURED LATENCY (authoritative) — the PR's own benchmarks
     (torch.profiler self-time for pre; bench_gpu_time for post).
  2. DERIVED HBM-BW SOL% — algorithmic-minimum bytes (each distinct input/output
     counted once) / measured latency, vs H100 SXM5 HBM3 peak (~3.35 TB/s).
     Valid only when the working set >> L2 (~50 MB); smaller shapes are
     latency-bound and reported as latency-only.
  3. STATIC RESOURCE USAGE — registers/thread, spills, smem/block via
     `cuobjdump --dump-resource-usage` on the JIT-compiled .so (no GPU needed).

Image (cached, built once):
  * nvidia/cuda:12.4.1-devel-ubuntu22.04 + torch (cu124) + flashinfer build deps
  * git clone flashinfer-ai/flashinfer, fetch `pull/3285/head`, submodules,
    `pip install -e .` (editable; mHC kernels JIT-compile on first call).

Runtime (each `modal run`):
  * The LOCAL study/source_code/PR_3285_files/ tree is mounted and overlaid onto
    the checkout (sha256 of each .cu printed to prove edits landed), then the
    benchmarks run; latency is parsed into a derived-bandwidth table.

Guardrails honored:
  * @app.function(..., timeout=900)   -> 15 min hard cap (compile + run)
  * subprocess.run(..., timeout=300)  -> 5 min cap per measurement phase
  * No git/gh operations; no local CUDA execution.

Usage:
  modal run profile_mhc_agent.py               # both kernels, full shape matrix
  modal run profile_mhc_agent.py --which post  # only mhc_post
  modal run profile_mhc_agent.py --which pre
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
# repo layout: <repo>/profiling/profile_mhc_agent.py  and  <repo>/study/...
PR_FILES_LOCAL = SCRIPT_DIR.parent / "study" / "source_code" / "PR_3285_files"

FLASHINFER_DIR = "/opt/flashinfer"            # cloned checkout inside container
PR_FILES_REMOTE = "/root/pr_files"            # runtime mount of local PR tree
CACHE_DIR = "/root/.cache/flashinfer"         # flashinfer's real JIT cache root (Volume-backed)

PR_NUMBER = 3285
TORCH_SPEC = "torch==2.5.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Subset of the local PR tree that maps onto real flashinfer paths.
# (We deliberately skip __init__.py / aot.py: the PR-head checkout already
#  contains the complete, wired-in versions; only kernel/binding files we may
#  edit need to win.)
OVERLAY_MAP = {
    "csrc/mhc": "csrc/mhc",
    "flashinfer/mhc.py": "flashinfer/mhc.py",
    "flashinfer/jit/mhc.py": "flashinfer/jit/mhc.py",
    "flashinfer/trace/templates/mhc.py": "flashinfer/trace/templates/mhc.py",
    "benchmarks/bench_mhc_pre_big_fuse.py": "benchmarks/bench_mhc_pre_big_fuse.py",
    "benchmarks/bench_mhc_post.py": "benchmarks/bench_mhc_post.py",
    "tests/mhc": "tests/mhc",
}

# The two kernels we optimize — hashed after overlay to confirm edits propagate.
CU_FILES = ["csrc/mhc/mhc_pre_big_fuse.cu", "csrc/mhc/mhc_post.cu"]

# Derived-bandwidth model (H100 SXM5 80GB HBM3 ~ 3.35 TB/s).
HBM_PEAK_GBPS = 3350.0
BF16, FP32, HC, MIX = 2, 4, 4, 24
# Working set must exceed ~L2 (50 MB) for derived BW% to reflect DRAM (else L2-bound).
L2_BYTES_THRESHOLD = 64e6

# Shape matrix: small N exposes the latency-bound regime (Sinkhorn serialization),
# N=4096 exposes the HBM-bound regime. H = DeepSeek-V4 Flash (4096) and Pro (7168).
SWEEP_SEQS = ["1", "128", "4096"]
SWEEP_HIDDENS = ["4096", "7168"]


def _bytes_post(n: int, h: int) -> int:
    # in: x(N*H) + residual(N*4*H); out: out(N*4*H); + post/comb mixes
    return BF16 * n * h * (1 + HC + HC) + n * HC * FP32 + n * HC * HC * FP32


def _bytes_pre_pure(n: int, h: int) -> int:
    # in: residual(N*4*H) + dot_mix(N*24) + sqrsum(N); out: layer(N*H) + post + comb
    return BF16 * n * h * (HC + 1) + n * MIX * FP32 + n * FP32 + n * HC * FP32 + n * HC * HC * FP32


def _bytes_pre_prenorm(n: int, h: int) -> int:
    # Algorithmic minimum (residual counted ONCE). The kernel currently reads
    # residual twice (sqrsum pass + reduction pass); the gap vs measured BW
    # quantifies that redundant HBM round-trip.
    return BF16 * n * h * (HC + 1) + n * MIX * FP32 + n * HC * FP32 + n * HC * HC * FP32

# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "ninja-build",
        "wget",
        "cuda-nsight-compute-12-4",  # provides `ncu`
    )
    .pip_install(
        TORCH_SPEC,
        extra_index_url=TORCH_INDEX,
    )
    .pip_install(
        # Pinned to match flashinfer's [build-system].requires exactly so that
        # `--no-build-isolation` uses a compatible toolchain. setuptools>=77 is
        # mandatory: flashinfer uses a PEP 639 SPDX `license = "Apache-2.0"`
        # string, which older (Modal-bundled) setuptools validators reject.
        "setuptools>=77",
        "packaging>=24",
        "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2",
        "ninja",
        "cmake",
        "wheel",
        "numpy",
        "pybind11",
        "requests",
        "tabulate",
        "filelock",
    )
    .env(
        {
            "FLASHINFER_ENABLE_AOT": "0",   # rely on JIT for mHC
            "TORCH_CUDA_ARCH_LIST": "9.0a",  # Hopper / H100
            "MAX_JOBS": "4",
        }
    )
    .run_commands(
        # Clone the real flashinfer, pin to the exact PR #3285 state, build editable.
        f"git clone https://github.com/flashinfer-ai/flashinfer.git {FLASHINFER_DIR}",
        f"cd {FLASHINFER_DIR} && git fetch origin pull/{PR_NUMBER}/head:pr{PR_NUMBER}",
        f"cd {FLASHINFER_DIR} && git checkout pr{PR_NUMBER}",
        f"cd {FLASHINFER_DIR} && git submodule update --init --recursive",
        f"cd {FLASHINFER_DIR} && pip install --no-build-isolation -e . -v",
        # No GPU needed for the editable JIT install (arch via TORCH_CUDA_ARCH_LIST);
        # mHC kernels JIT-compile at runtime inside the H100 function.
    )
    # Modal's host driver is R580 / CUDA 13. The CUDA-12.4 toolkit's bundled
    # ncu (2024.1.1) segfaults in cuDevicePrimaryCtxRetain on that driver, so
    # install a newer Nsight Compute. Appended as a late layer to preserve the
    # cached torch + flashinfer build layers. Fallback across CUDA versions.
    .run_commands(
        "apt-get update && ("
        "apt-get install -y cuda-nsight-compute-13-0 || "
        "apt-get install -y cuda-nsight-compute-12-9 || "
        "apt-get install -y cuda-nsight-compute-12-8 || "
        "apt-get install -y cuda-nsight-compute-12-6)"
    )
    # Late env layer (preserves cached torch/flashinfer/nsight layers): ask the
    # GPU runtime to mount the FULL driver library set, including the profiling
    # library Nsight Compute needs. Default `compute,utility` omits it, which
    # surfaces as "Failed to initialize the profiler: LibraryNotLoaded".
    .env(
        {
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "NVIDIA_VISIBLE_DEVICES": "all",
        }
    )
    # pytest correctness gate (late small layer; preserves the cached build).
    .pip_install("pytest")
    # Runtime mount of the local PR tree (copy=False => fast, no image rebuild on edit).
    .add_local_dir(str(PR_FILES_LOCAL), PR_FILES_REMOTE, copy=False)
)

app = modal.App("mhc-profiler", image=image)


# --------------------------------------------------------------------------- #
# Helpers (run inside the container)
# --------------------------------------------------------------------------- #
def _sh(cmd: list[str], *, timeout: int, cwd: str, env: dict, title: str) -> tuple[int, str]:
    """Run a subprocess, stream combined output, honor a hard timeout.

    Returns (returncode, captured_text). On timeout the process is killed
    (guardrail) and returncode 124 is returned with whatever was captured.
    """
    print(f"\n{'=' * 78}\n>>> {title}\n>>> $ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(proc.stdout, flush=True)
        print(f">>> [{title}] exit_code={proc.returncode}", flush=True)
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        # Guardrail: kill the runaway process and log the failure, keep going.
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        print(partial, flush=True)
        print(f">>> [{title}] TIMED OUT after {timeout}s — process killed.", flush=True)
        return 124, partial


# Bench output line parsers.
_PRE_RE = re.compile(
    r"^\s*(pure|prenorm)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$"
)
_POST_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")


def _bw_line(name: str, n: int, h: int, lat_us: float, nbytes: int) -> str:
    eff_gbps = nbytes / lat_us / 1e3  # bytes / (lat_us*1e-6) / 1e9
    ws_mb = nbytes / 1e6
    if nbytes >= L2_BYTES_THRESHOLD:
        pct = eff_gbps / HBM_PEAK_GBPS * 100.0
        bw = f"{eff_gbps:9.1f} {pct:7.1f}"
    else:
        bw = f"{'(<=L2)':>9} {'lat-bd':>7}"
    return f"{name:>24} {n:6d} {h:6d} {lat_us:10.2f} {ws_mb:9.1f} {bw}"


def _report_bandwidth(pre_txt: str, post_txt: str) -> None:
    print(f"\n{'#' * 78}\n# DERIVED HBM BANDWIDTH (algorithmic-min bytes / measured latency)")
    print(f"# H100 SXM5 HBM3 peak = {HBM_PEAK_GBPS:.0f} GB/s; BW% valid only for "
          f"working set > {L2_BYTES_THRESHOLD/1e6:.0f} MB\n{'#' * 78}")
    print(f"{'kernel':>24} {'N':>6} {'H':>6} {'lat_us':>10} {'WS_MB':>9} {'GB/s':>9} {'%peak':>7}")
    print("-" * 78)
    for line in pre_txt.splitlines():
        m = _PRE_RE.match(line)
        if not m:
            continue
        mode, n, h = m.group(1), int(m.group(2)), int(m.group(3))
        lat = float(m.group(8))
        nbytes = _bytes_pre_pure(n, h) if mode == "pure" else _bytes_pre_prenorm(n, h)
        print(_bw_line(f"mhc_pre_{mode}", n, h, lat, nbytes))
    for line in post_txt.splitlines():
        m = _POST_RE.match(line)
        if not m:
            continue
        n, h = int(m.group(1)), int(m.group(2))
        lat = float(m.group(4))  # median us
        print(_bw_line("mhc_post", n, h, lat, _bytes_post(n, h)))
    print("-" * 78, flush=True)


def _find_cuobjdumps() -> list[str]:
    cands: list[str] = []
    w = shutil.which("cuobjdump")
    if w:
        cands.append(w)
    cands += [str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/cuobjdump"))]
    cands += [str(p) for p in sorted(Path("/usr/local/lib").rglob("nvidia/*/bin/cuobjdump"))]
    # de-dup, keep order
    seen, out = set(), []
    for c in cands:
        if c not in seen and Path(c).exists():
            seen.add(c)
            out.append(c)
    return out


def _dump_resource_usage(env: dict) -> None:
    """Per-kernel registers/thread, spills, smem via cuobjdump on the JIT .so."""
    print(f"\n{'#' * 78}\n# STATIC RESOURCE USAGE (cuobjdump --dump-resource-usage)\n{'#' * 78}")

    # Where did flashinfer actually put the JIT artifacts?
    _sh([sys.executable, "-c",
         "from flashinfer.jit import env as e;"
         "print('FLASHINFER_WORKSPACE_DIR=', e.FLASHINFER_WORKSPACE_DIR);"
         "print('FLASHINFER_JIT_DIR=', e.FLASHINFER_JIT_DIR)"],
        timeout=60, cwd=FLASHINFER_DIR, env=env, title="flashinfer JIT dirs")

    # Locate the compiled mHC shared object wherever it landed.
    _, out = _sh(
        ["bash", "-lc",
         "for d in /cache /root/.cache /root/.flashinfer ~/.cache /tmp /opt; do "
         "find \"$d\" -name '*.so' 2>/dev/null; done | grep -i mhc | sort -u"],
        timeout=120, cwd=FLASHINFER_DIR, env=env, title="locate mHC .so",
    )
    sos = [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".so")]
    if not sos:
        print("  no mhc .so found; cannot dump resource usage")
        return
    target = sos[0]
    print("  target .so:", target)
    for cu in _find_cuobjdumps():
        rc, dump = _sh([cu, "--dump-resource-usage", target], timeout=120,
                       cwd=FLASHINFER_DIR, env=env, title=f"cuobjdump ({cu})")
        if rc == 0 and "REG" in dump.upper():
            return
    print("  (cuobjdump could not parse the .so with any available toolchain)")


def _run_tests(env: dict) -> int:
    """Correctness gate: run the PR's reference-comparison tests on the kernels."""
    print(f"\n{'#' * 78}\n# CORRECTNESS GATE (pytest tests/mhc vs reference)\n{'#' * 78}")
    rc, _ = _sh([sys.executable, "-m", "pytest", "-q", "--no-header", "tests/mhc"],
                timeout=300, cwd=FLASHINFER_DIR, env=env, title="pytest tests/mhc")
    print(f">>> CORRECTNESS: {'PASS' if rc == 0 else 'FAIL'} (rc={rc})", flush=True)
    return rc


def _overlay_pr_files() -> None:
    """Copy the locally-mounted PR tree over the cloned flashinfer checkout."""
    import hashlib

    print(f"\n{'#' * 78}\n# Overlaying local PR_3285_files -> {FLASHINFER_DIR}\n{'#' * 78}")
    src_root = Path(PR_FILES_REMOTE)
    dst_root = Path(FLASHINFER_DIR)
    for rel_src, rel_dst in OVERLAY_MAP.items():
        src = src_root / rel_src
        dst = dst_root / rel_dst
        if not src.exists():
            print(f"  [skip] {rel_src} (not present locally)")
            continue
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(src)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)
            print(f"  [dir ] {rel_src} -> {rel_dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  [file] {rel_src} -> {rel_dst}")

    print("\n# Overlaid kernel fingerprints (sha256, first 16 hex):")
    for rel in CU_FILES:
        p = dst_root / rel
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            print(f"  {h}  {rel}")
        else:
            print(f"  <missing>          {rel}")


# --------------------------------------------------------------------------- #
# Remote function
# --------------------------------------------------------------------------- #
@app.function(gpu="H100", timeout=900)
def profile(which: str = "both", run_tests: bool = True) -> None:
    env = dict(os.environ)
    py = sys.executable

    # ---- environment banner ------------------------------------------------ #
    _sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
        timeout=60, cwd=FLASHINFER_DIR, env=env, title="GPU")
    _sh(["bash", "-lc", "which nvcc && nvcc --version | tail -3"],
        timeout=60, cwd=FLASHINFER_DIR, env=env, title="nvcc")

    # ---- overlay local edits ----------------------------------------------- #
    _overlay_pr_files()

    do_pre = which in ("both", "pre")
    do_post = which in ("both", "post")
    pre_bench = "benchmarks/bench_mhc_pre_big_fuse.py"
    post_bench = "benchmarks/bench_mhc_post.py"

    # ---- correctness gate (also triggers the JIT compile) ------------------- #
    if run_tests:
        _run_tests(env)

    # ---- latency sweep (also JIT-compiles on first call) -------------------- #
    # cold-L2 (default rotate mode) makes large-N reads come from DRAM so the
    # derived bandwidth reflects true HBM traffic; --max-rotations bounds setup.
    pre_txt = post_txt = ""
    if do_pre:
        _, pre_txt = _sh(
            [py, pre_bench,
             "--sequence-lengths", *SWEEP_SEQS,
             "--hidden-sizes", *SWEEP_HIDDENS,
             "--num-splits", "1",
             "--max-rotations", "64"],
            timeout=300, cwd=FLASHINFER_DIR, env=env,
            title="LATENCY SWEEP: mhc_pre_big_fuse (cold-L2)",
        )
    if do_post:
        _, post_txt = _sh(
            [py, post_bench,
             "--sequence-lengths", *SWEEP_SEQS,
             "--hidden-sizes", *SWEEP_HIDDENS],
            timeout=300, cwd=FLASHINFER_DIR, env=env,
            title="LATENCY SWEEP: mhc_post",
        )

    # ---- derived HBM-BW SOL% + static register/smem/spill report ------------ #
    _report_bandwidth(pre_txt, post_txt)
    _dump_resource_usage(env)

    print("\n>>> profiling run complete.", flush=True)


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(which: str = "both", run_tests: bool = True) -> None:
    profile.remote(which=which, run_tests=run_tests)
