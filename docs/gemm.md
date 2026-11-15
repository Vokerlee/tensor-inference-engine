# BF16 matrix product

`tensor_inference_engine::gemm` computes a two-dimensional `A @ B` with BF16
operands, FP32 accumulation, and either BF16 or FP32 row-major output. The
public header is `include/tensor_inference_engine/gemm.hpp`.

Inputs may be row-major or column-major with a positive leading dimension,
including pitched storage. Dimensions and leading dimensions must fit int32.
The caller owns device memory, current device, stream, and synchronization.
Output must not overlap either input. The API does not allocate or synchronize.

The implementation instantiates CUTLASS SM80 Tensor Core kernels, compiled for
the selected CUDA architectures (SM89 by default). Aligned matrices use
vectorized operand/epilogue accesses; a scalar-access variant handles tails and
unaligned leading dimensions. A simple shape heuristic uses 64x64x32 tiles
when either output dimension is at most 256, and
128x128x32 tiles otherwise. Both use FP32 accumulation; this is a baseline
heuristic, not an autotuned optimum. Zero K writes zero; zero M or N performs
no device work.

Build and run the native checks with the root README commands. `gemm_native`
compares both output types and all layout pairs against a CPU double-precision
reference built from exactly represented binary fractions. It also checks
empty reductions, tails, pitched output padding and invalid descriptors.

This is a GPU numerical baseline. Tensor Core instruction/reduction ordering
does not promise bitwise equality with another library or emulate a target
accelerator's intermediate rounding. No quantization policy, arbitrary
element-strided input, batching, convolution, or autograd is included here.
