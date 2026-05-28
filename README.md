
# DeepSeek V4 — MHC & GPU Kernel Study

This repository exists to contribute to FlashInfer's DeepSeek V4 work (see the issue linked below) by studying the model-level optimizations and porting the reference preprocessing kernels into CUDA for FlashInfer's codebase. It is a personal research workspace and a place to document and showcase implementation notes and experiments.

Primary objective
- Study the DeepSeek V4 optimizations and how they are implemented in FlashInfer, and implement those preprocessing kernels and performance improvements on CUDA.

Key references (in-repo)
- [DeepSeek V4 paper / notes](DeepSeek_V4.pdf)
- [MHC theory & notes](study/llm_theory/mhc.md)
- [GPU optimizations (split-K, tiling)](study/gpu_theory/)
- [Reference code: mhc_pre.py](study/source_code/mhc_pre.py)
- [Progress timeline](timeline.md)
- [AI-generated plan](plan.md) made with Gemini 3.1 pro
- [Diagram: hyper-connection](study/media/hc_diagram.png)

Context
- This repo is maintained as a personal study / research workspace to explore DeepSeek V4's MHC design and the GPU kernels that support it. Work here is intended to inform contributions to FlashInfer (see issue: https://github.com/flashinfer-ai/flashinfer/issues/3346).

Getting started
1. Read the model/design notes: [study/llm_theory/mhc.md](study/llm_theory/mhc.md).
2. Review GPU optimization strategies: [study/gpu_theory/](study/gpu_theory).
3. Inspect the reference implementation: [study/source_code/mhc_pre.py](study/source_code/mhc_pre.py).
4. Follow progress and experiments in [timeline.md](timeline.md) and the high-level [plan.md](plan.md).

Suggested next steps
- Prototype a `cuda/` sandbox and implement the simplest mhc preprocessing kernel (GEMM + sqr-sum epilogue) for parity tests against the Python reference.
- Add microbenchmarks and small test vectors to validate correctness and measure performance.
- Document observations and kernel-design tradeoffs in `study/`.

Notes
- This repository is for personal study and demonstration; it is not intended as a multi-contributor project. The README intentionally omits a "Contributing" section.

License
- See the [LICENSE](LICENSE) file in the repo root.


