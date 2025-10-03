
# CUDA starter — notebook summary

This repository contains a hands-on Jupyter notebook: `notebooks/cuda-starter.ipynb`. The notebook demonstrates writing, compiling, and running CUDA kernels from Python using PyTorch's C++/CUDA extension utilities. The goal is to show how algorithm implementations evolve from easy-to-read Python versions to high-performance CUDA kernels.

## Overview

- Progressive implementations: pure Python -> simulated kernels/blocks in Python -> inline C++/CUDA kernels loaded with `torch.utils.cpp_extension`.
- Worked examples: RGB→grayscale image conversion and matrix multiplication (matmul).
- Emphasis on mapping work to GPU threads/blocks and validating correctness vs. PyTorch's built-ins.

## What I learned (concise)

- How to set up and use a CUDA-enabled PyTorch environment and load images with `torchvision`.
- How kernels map to threads and blocks (1D, 2D indexing), and the limits to consider (max threads, block dims).
- How to write inline CUDA C++ kernels, compile them at runtime, and call them from Python.
- Basic debugging and safety patterns: input/contiguity checks, `C10_CUDA_KERNEL_LAUNCH_CHECK()`, and `CUDA_LAUNCH_BLOCKING=1` for deterministic failures.
- Performance trade-offs between pure Python implementations and CUDA kernels; how to validate correctness using `torch.isclose` and CPU/PyTorch comparisons.

## Notebook highlights (selected snippets)

- RGB → Grayscale
  - `rgb2grey_py`: Python flatten-based conversion for clarity.
  - `rgb2grey_pyk` / `rgb2grey_pybk`: Python-simulated kernels illustrating single-thread and block behavior.
  - `rgb_to_grayscale` (CUDA): inline C++/CUDA kernel with 1D and later 2D grid/block indexing.

- Matrix multiplication (matmul)
  - Naive triple-loop Python matmul for clarity and correctness checks.
  - 2D block-simulated kernel and a simple CUDA kernel using `dim3` blocks and thread-per-block configurations.
  - Direct comparisons to `m1 @ m2` (PyTorch) for correctness and timing.

## How to run locally (notes)

1. Ensure you have a CUDA-capable GPU and matching CUDA toolkit installed.
2. Create a Python environment and install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib notebook
```

3. Launch the notebook:

```bash
jupyter notebook notebooks/cuda-starter.ipynb
```

4. Runtime tips:

- Set `CUDA_LAUNCH_BLOCKING=1` in the environment (or run the notebook cell that sets it) while developing to make errors deterministic.
- Make tensors contiguous and move them to GPU with `.contiguous().cuda()` before passing to CUDA kernels.
- When editing CUDA/C++ code in the notebook, the inline loader recompiles; watch the verbose output for compile/link errors.

## Next steps & ideas

- Implement shared-memory tiled matmul to reduce global memory traffic and improve performance.
- Profile with Nsight Systems / Nsight Compute or `torch.cuda.profiler` to locate bottlenecks.
- Explore mixed-precision and tensor cores on recent NVIDIA GPUs for faster matmuls.
- Add more image-processing kernels (convolution, reduction operations) and compare to cuDNN/cuBLAS where relevant.

## References

1. [Visualizing attention (original repo topic)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
2. CUDA / lecture reference used in the notebook: "Getting Started With CUDA for Python Programmers" (lecture referenced in the notebook).


