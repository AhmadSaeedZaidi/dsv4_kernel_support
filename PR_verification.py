"""
run_mhc_pr_validation.py — Automatic validation script for FlashInfer MHC pull requests.

This script executes a two-phase benchmarking methodology on a remote Modal H100 
instance to provide an exact side-by-side performance delta.
  Phase 1: Validates and profiles the original upstream branch.
  Phase 2: Overwrites the target kernels with optimized local files, recompiles 
           inline, and profiles the updated configuration.
"""

import os
import subprocess
import sys
from pathlib import Path
import modal

app = modal.App("flashinfer-mhc-pr-validator")

LOCAL_MHC_DIR = Path(__file__).resolve().parent / "study" / "source_code" / "PR_3285_files"
FLASHINFER_DIR = "/opt/flashinfer"
PR_NUMBER = 3285

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
        "cuda-nsight-compute-12-4",
    )
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
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
        "pytest",
        "filelock",
    )
    .env(
        {
            "FLASHINFER_ENABLE_AOT": "0",
            "TORCH_CUDA_ARCH_LIST": "9.0a",
            "MAX_JOBS": "4",
        }
    )
    .run_commands(
        f"git clone https://github.com/flashinfer-ai/flashinfer.git {FLASHINFER_DIR}",
        f"cd {FLASHINFER_DIR} && git fetch origin pull/{PR_NUMBER}/head:pr{PR_NUMBER}",
        f"cd {FLASHINFER_DIR} && git checkout pr{PR_NUMBER}",
        f"cd {FLASHINFER_DIR} && git submodule update --init --recursive",
        f"cd {FLASHINFER_DIR} && pip install --no-build-isolation -e . -v",
    )
    .add_local_dir(str(LOCAL_MHC_DIR), remote_path="/root/optimized_code", copy=False)
)

def _sh(cmd: list[str], *, timeout: int = 300, cwd: str, env: dict, title: str) -> tuple[int, str]:
    """Execute subprocess, stream output, and enforce timeouts."""
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
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        print(partial, flush=True)
        print(f">>> [{title}] EXCEEDED RUNTIME LIMIT {timeout}s — Terminated.", flush=True)
        return 124, partial
    except Exception as e:
        print(f">>> [{title}] FAILED TO EXECUTE: {e}", flush=True)
        return 1, str(e)

@app.function(gpu="h100", timeout=1200)
def run_validation():
    env = dict(os.environ)
    py = sys.executable
    
    # -------------------------------------------------------------------------
    # PHASE 1: Baseline Profiling
    # -------------------------------------------------------------------------
    print("\n" + "#" * 78)
    print("# PHASE 1: ORIGINAL UPSTREAM BASELINE")
    print("#" * 78)
    
    _sh([py, "-m", "pytest", "-v", "tests/mhc/"], 
        cwd=FLASHINFER_DIR, env=env, timeout=300, title="Correctness Gate (Baseline)")
        
    _, baseline_post = _sh([py, "benchmarks/bench_mhc_post.py"], 
        cwd=FLASHINFER_DIR, env=env, timeout=300, title="Benchmark Baseline Post Mapping")
        
    _, baseline_pre = _sh([py, "benchmarks/bench_mhc_pre_big_fuse.py"], 
        cwd=FLASHINFER_DIR, env=env, timeout=300, title="Benchmark Baseline Pre Big Fuse")
    
    # -------------------------------------------------------------------------
    # PHASE 2: Apply Local Modifications and Profile
    # -------------------------------------------------------------------------
    print("\n" + "#" * 78)
    print("# PHASE 2: OPTIMIZED CONFIGURATION")
    print("#" * 78)
    
    import shutil
    src_file = "/root/optimized_code/csrc/mhc/mhc_pre_big_fuse.cu"
    dst_file = f"{FLASHINFER_DIR}/csrc/mhc/mhc_pre_big_fuse.cu"
    print(f">>> Copying optimized kernel from {src_file} to {dst_file}")
    shutil.copy2(src_file, dst_file)
    
    # Executing the test suite triggers JIT to recompile the newly injected .cu file
    _sh([py, "-m", "pytest", "-v", "tests/mhc/"], 
        cwd=FLASHINFER_DIR, env=env, timeout=300, title="Correctness Gate (Optimized)")
        
    _, optimized_post = _sh([py, "benchmarks/bench_mhc_post.py"], 
        cwd=FLASHINFER_DIR, env=env, timeout=300, title="Benchmark Optimized Post Mapping")
        
    _, optimized_pre = _sh([py, "benchmarks/bench_mhc_pre_big_fuse.py"], 
        cwd=FLASHINFER_DIR, env=env, timeout=300, title="Benchmark Optimized Pre Big Fuse")
    
    # -------------------------------------------------------------------------
    # Final Output Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("=================== PERFORMANCE VALIDATION SUMMARY ===================")
    print("=" * 78)
    
    print("\n[MHC POST: UPSTREAM BASELINE]\n" + "-" * 78)
    print(baseline_post)
    print("\n[MHC POST: OPTIMIZED PATH]\n" + "-" * 78)
    print(optimized_post)
    
    print("\n[MHC PRE: UPSTREAM BASELINE]\n" + "-" * 78)
    print(baseline_pre)
    print("\n[MHC PRE: OPTIMIZED PATH]\n" + "-" * 78)
    print(optimized_pre)

@app.local_entrypoint()
def main():
    run_validation.remote()