### Phase 1: Understand the Ground Truth (The Math)

Before writing any CUDA, you need to understand exactly what bits are moving where.

1.  **Find the Reference:** Look at the SGLang codebase you linked (`sglang/srt/layers/mhc.py`). Find the `hc_pre_torch_impl` function.
2.  **Write a Standalone Python Script:** Extract that PyTorch implementation into a single `test_mhc_pre.py` file. Create random tensors (matching DeepSeek V4 dimensions), run the Torch version, and save the outputs.
3.  **Understand the IO:** \* **Inputs:** Input tensor (X), RMSNorm weights, GEMM weights.
    
    -   **Outputs:** GEMM output (Y), and a vector containing the sum of squares of Y across the hidden dimension.

### Phase 2: The Hopper (SM90) Strategy

Writing kernels for SM90 is fundamentally different from older architectures (Ampere/Ada). You do not write standard CUDA threads mapping to memory loops. You use **CuTe** and **CUTLASS 3.x**.

1.  **Learn the Hardware:** You will be utilizing **TMA** (Tensor Memory Accelerator) for async memory fetches and **WGMMA** (Warp-Group Matrix Multiply Accumulate) for the math.
2.  **Epilogue Visitor Tree (EVT):** Because this kernel is a GEMM with a custom prefix (RMSNorm) and suffix (SqrSum), the standard way to write this in CUTLASS 3.x is using an **EVT**. This allows you to use a highly optimized SM90 GEMM core and just write custom templates for what happens immediately before and after the math while the data is still hot in the registers.
3.  **Study FlashInfer's Codebase:** Look at how FlashInfer handles SM90 GEMMs. Check their `include/flashinfer/` directory. See if they are wrapping CUTLASS 3.x or if they have their own CuTe abstractions for Hopper.

### Phase 3: Setting Up Your Modal Sandbox

Don't try to integrate into FlashInfer's massive C++ build system immediately. Compile times will eat your Modal credits and your sanity.

1.  **Create a Minimal Modal App:** Write a Modal script that requests an H100 (`gpu="h100"`), installs PyTorch, and uses `torch.utils.cpp_extension.load_inline` or Pybind11 to compile a single `.cu` file.
2.  **Write a Naive CUDA Kernel:** First, write a completely naive, slow CUDA kernel (no TMA, no WGMMA) just to get the math right and pass the `torch.allclose` test against your Phase 1 script.
3.  **Write the SM90 Kernel:** Evolve your naive kernel using CUTLASS/CuTe to target TMA and WGMMA. Profile it using Nsight Compute (`ncu`) on Modal to ensure it's actually faster than the Torch fallback.

### Phase 4: Integration into FlashInfer

Once you have a blazing fast, tested standalone `.cu` file, _then_ you clone FlashInfer. Look at their dispatch mechanism. You will need to wrap your kernel in their C++ templates, add it to their Pybind11 bindings, and add unit tests in their `tests/` directory.

### When and How to Reach Out to Po-Han [(po han linkedin)](https://www.linkedin.com/in/phuang17/)

**Checkpoint 1: The Architecture Check (Very Soon)**

-   _Action:_ After you have extracted the PyTorch reference, understood the math, and looked at FlashInfer's repo.
-   _What to say:_ "Hi Po-Han, I have Modal H100s and I'm taking a swing at the `MHC-PRE-GEMM` SM90 kernel for #3346. I understand the math (PreNorm + GEMM + SqrSum Epilogue). Before I write the CUDA, what is FlashInfer's preferred approach for custom SM90 GEMMs right now? Should I use CUTLASS 3 EVT, or do you have a specific raw CuTe template in the codebase I should adapt?"
-   _Why:_ This saves you from spending 20 hours writing a CUTLASS kernel if he actually wants it written in raw CuTe to match his internal roadmap.

**Checkpoint 2: The Integration Block**

-   _Action:_ You have a standalone kernel running on Modal that produces correct results and is fast, but you don't understand FlashInfer's templating/dispatch system to hook it into the Python API.
-   _What to say:_ "I have a working SM90 kernel for `MHC-PRE-GEMM` that passes parity with the SGLang Torch fallback and profiles well. I'm having trouble figuring out the correct file structure and dispatch macros to expose it to the Python frontend. Could you point me to a similar PR or the right files to modify?"

**Checkpoint 3: The PR Review**

-   _Action:_ You've integrated it, and it works. Open a Draft PR.
-   _What to say:_ "Draft PR for MHC-PRE-GEMM (SM90). Tests pass on H100. Let me know what you think of the implementation and if any formatting/style changes are needed."