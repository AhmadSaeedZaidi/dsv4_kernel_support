"""
pr_bench_mhc.py — H100 head-to-head benchmark for the mHC pre big-fuse change.

Flow (single remote H100 container):
  1. Clone flashinfer @ PR #3285 (pristine) and benchmark the ORIGINAL kernel.
  2. Overlay the local optimized `csrc/mhc/mhc_pre_big_fuse.cu`, drop the JIT
     cache so it recompiles, run the repo's pre-commit hook on the changed file,
     and the reference correctness tests.
  3. Re-benchmark the optimized kernel and print original-vs-optimized tables.

The only file this PR changes is `csrc/mhc/mhc_pre_big_fuse.cu`; the optimization
is gated to `__CUDA_ARCH__ == 900`, so non-Hopper builds are byte-identical to
upstream and the comparison below is meaningful only on H100 (sm_90).

  modal run pr_bench_mhc.py

Guardrails: gpu="H100", timeout=900; every subprocess.run(..., timeout=300).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
PR_FILES_LOCAL = SCRIPT_DIR.parent / "study" / "source_code" / "PR_3285_files"
FLASHINFER_DIR = "/opt/flashinfer"
PR_FILES_REMOTE = "/root/pr_files"
PR_NUMBER = 3285
TORCH_SPEC = "torch==2.5.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
KERNEL_REL = "csrc/mhc/mhc_pre_big_fuse.cu"  # the only file this PR changes

# pre: warm-L2 (matches the PR's "enable l2 cache" table); post: default sweep.
PRE_SEQS = ["1", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096", "8192"]
POST_SEQS = ["1", "32", "64", "128", "256", "512", "1024", "2048", "4096", "8192"]
HIDDENS = ["4096", "7168"]

# Image: identical recipe to the profiler (so the expensive flashinfer clone +
# editable install layers are reused from cache), plus pytest + pre-commit.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "wget",
                 "cuda-nsight-compute-12-4")
    .pip_install(TORCH_SPEC, extra_index_url=TORCH_INDEX)
    .pip_install(
        "setuptools>=77", "packaging>=24",
        "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2",
        "ninja", "cmake", "wheel", "numpy", "pybind11", "requests", "tabulate", "filelock",
    )
    .env({"FLASHINFER_ENABLE_AOT": "0", "TORCH_CUDA_ARCH_LIST": "9.0a", "MAX_JOBS": "4"})
    .run_commands(
        f"git clone https://github.com/flashinfer-ai/flashinfer.git {FLASHINFER_DIR}",
        f"cd {FLASHINFER_DIR} && git fetch origin pull/{PR_NUMBER}/head:pr{PR_NUMBER}",
        f"cd {FLASHINFER_DIR} && git checkout pr{PR_NUMBER}",
        f"cd {FLASHINFER_DIR} && git submodule update --init --recursive",
        f"cd {FLASHINFER_DIR} && pip install --no-build-isolation -e . -v",
    )
    .run_commands(
        "apt-get update && (apt-get install -y cuda-nsight-compute-13-0 || "
        "apt-get install -y cuda-nsight-compute-12-9 || true)"
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "NVIDIA_VISIBLE_DEVICES": "all"})
    .pip_install("pytest", "pre-commit")
    .add_local_dir(str(PR_FILES_LOCAL), PR_FILES_REMOTE, copy=False)
)

app = modal.App("mhc-pr-bench", image=image)


def _sh(cmd: list[str], *, timeout: int, env: dict, title: str, check: bool = False) -> tuple[int, str]:
    print(f"\n{'=' * 78}\n>>> {title}\n>>> $ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    try:
        p = subprocess.run(cmd, cwd=FLASHINFER_DIR, env=env, timeout=timeout, check=False,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(p.stdout, flush=True)
        print(f">>> [{title}] exit_code={p.returncode}", flush=True)
        return p.returncode, p.stdout or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        print(out, flush=True)
        print(f">>> [{title}] TIMED OUT after {timeout}s — killed.", flush=True)
        return 124, out


_PRE_RE = re.compile(r"^\s*(pure|prenorm)\s+(\d+)\s+(\d+)\s+\d+\s+\w+\s+\d+\s+\d+\s+([\d.]+)\s*$")
_POST_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+\d+\s+([\d.]+)\s+[\d.]+\s*$")


def _parse_pre(txt: str) -> dict:
    out: dict = {}
    for ln in txt.splitlines():
        m = _PRE_RE.match(ln)
        if m:
            out[(m.group(1), int(m.group(3)), int(m.group(2)))] = float(m.group(4))  # (mode,H,N)
    return out


def _parse_post(txt: str) -> dict:
    out: dict = {}
    for ln in txt.splitlines():
        m = _POST_RE.match(ln)
        if m:
            out[(int(m.group(2)), int(m.group(1)))] = float(m.group(3))  # (H,N) -> median us
    return out


def _emit(title: str, axis: str, seqs: list[str], orig: dict, opt: dict, key) -> None:
    print(f"\n{title}")
    for h in (4096, 7168):
        print(f"\nH={h}")
        print(f"{axis:<6} {'original':>11} {'optimized':>11} {'speedup':>9}")
        for s in seqs:
            n = int(s)
            a, b = orig.get(key(h, n)), opt.get(key(h, n))
            if a is None or b is None:
                continue
            print(f"{n:<6} {a:>11.3f} {b:>11.3f} {a / b:>8.2f}x")


@app.function(gpu="H100", timeout=900)
def compare() -> None:
    env = dict(os.environ)
    py = sys.executable
    pre_b, post_b = "benchmarks/bench_mhc_pre_big_fuse.py", "benchmarks/bench_mhc_post.py"
    pre_args = ["--sequence-lengths", *PRE_SEQS, "--hidden-sizes", *HIDDENS,
                "--num-splits", "1", "--no-cold-l2-cache"]
    post_args = ["--sequence-lengths", *POST_SEQS, "--hidden-sizes", *HIDDENS]

    _sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
        timeout=60, env=env, title="GPU")

    # ---- 1. ORIGINAL (pristine PR-head checkout) --------------------------- #
    _, pre_orig_txt = _sh([py, pre_b, *pre_args], timeout=300, env=env, title="ORIGINAL: pre")
    _, post_orig_txt = _sh([py, post_b, *post_args], timeout=300, env=env, title="ORIGINAL: post")

    # ---- 2. overlay optimized kernel + drop JIT cache ---------------------- #
    for f in (Path(PR_FILES_REMOTE) / "csrc/mhc").glob("*.cu"):
        shutil.copy2(f, Path(FLASHINFER_DIR) / "csrc/mhc" / f.name)
    _sh(["bash", "-lc", "rm -rf /root/.cache/flashinfer/*/*/cached_ops/mhc"],
        timeout=60, env=env, title="drop JIT cache for mhc")

    # ---- pre-commit on the changed file (capture clang-format delta) ------- #
    _sh(["bash", "-lc",
         f"cp {KERNEL_REL} /tmp/before.cu; "
         f"pre-commit run --files {KERNEL_REL} || true; "
         f"echo '--- clang-format diff (empty == compliant) ---'; "
         f"diff -u /tmp/before.cu {KERNEL_REL} || true"],
        timeout=300, env=env, title="pre-commit (clang-format) on changed file")

    # ---- correctness gate -------------------------------------------------- #
    rc, _ = _sh([py, "-m", "pytest", "-q", "--no-header", "tests/mhc"],
                timeout=300, env=env, title="correctness: pytest tests/mhc")
    print(f">>> CORRECTNESS: {'PASS' if rc == 0 else 'FAIL'} (rc={rc})", flush=True)

    # ---- 3. OPTIMIZED ------------------------------------------------------ #
    _, pre_opt_txt = _sh([py, pre_b, *pre_args], timeout=300, env=env, title="OPTIMIZED: pre")
    _, post_opt_txt = _sh([py, post_b, *post_args], timeout=300, env=env, title="OPTIMIZED: post")

    # ---- comparison tables ------------------------------------------------- #
    pre_o, pre_n = _parse_pre(pre_orig_txt), _parse_pre(pre_opt_txt)
    post_o, post_n = _parse_post(post_orig_txt), _parse_post(post_opt_txt)
    print(f"\n{'#' * 78}\n# mHC original vs optimized (H100 sm_90a) — latency (us)\n{'#' * 78}")
    _emit("mHC pre big fuse [BigFuse], enable l2 cache", "M/S", PRE_SEQS,
          {(h, n): v for (m, h, n), v in pre_o.items() if m == "pure"},
          {(h, n): v for (m, h, n), v in pre_n.items() if m == "pure"}, lambda h, n: (h, n))
    _emit("mHC pre big fuse [BigFuse+Prenorm], enable l2 cache", "M/S", PRE_SEQS,
          {(h, n): v for (m, h, n), v in pre_o.items() if m == "prenorm"},
          {(h, n): v for (m, h, n), v in pre_n.items() if m == "prenorm"}, lambda h, n: (h, n))
    _emit("mHC post mapping (unchanged by this PR — parity check)", "N", POST_SEQS,
          post_o, post_n, lambda h, n: (h, n))
    print("\n>>> comparison complete.", flush=True)


@app.local_entrypoint()
def main() -> None:
    compare.remote()
