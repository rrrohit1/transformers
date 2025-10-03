# Transformers & CUDA Projects

## CUDA Starter Notebook

A hands-on exploration of CUDA programming with PyTorch:

- Progressive implementations: pure Python → simulated kernels → inline C++/CUDA kernels
- Worked examples: RGB → grayscale conversion and matrix multiplication
- Core concepts: thread/block mapping, launch configs, memory management

## Flash Attention Explained

FlashAttention is a faster way to compute attention in Transformer models. It dramatically reduces memory use and improves speed by processing attention in small blocks that fit in fast GPU memory.

### The Standard Attention Formula

Normally, attention is computed as:

$$
\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

### The Problem

For sequence length $N$:

- The attention score matrix $QK^T$ is $N \times N$
- With long sequences (4k, 16k tokens), this becomes huge
- Results in high memory usage and computation time

### The Key Trick

Instead of building the full $N \times N$ score matrix, FlashAttention:

1. **Splits into Blocks**: Divides data into tiles that fit in GPU SRAM
2. **Streams Processing**: Computes block-by-block with on-the-fly softmax
3. **Saves Memory**: Never stores the full attention matrix
4. **Maintains Accuracy**: Unlike approximations (e.g., Linformer), results match standard attention

### Why Is It Faster?

Modern GPU bottlenecks:

- Moving data between HBM (slow) and SRAM (fast) is expensive
- FlashAttention trades some recomputation for much less memory traffic
- IO-aware design: optimized for memory hierarchy, not just compute ops

### Benefits

- **Memory Efficient**: Much lower memory use enables longer sequences
- **Faster**: 2–4× speedup on A100/H100 GPUs
- **Exact**: Same results as standard attention

### In Simple Words

Think of FlashAttention as computing attention in small "flashes" (tiles) inside the GPU's fast memory. Instead of storing one huge attention matrix, it works with small pieces that fit in fast memory, trading a bit of recomputation for much less slow memory access.