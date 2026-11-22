# Grouped representation reference

`src/selection_reference.py` is an optional Torch numerical reference. It has
no model, dataset or research-workflow dependency. Load it with `importlib`
when comparing activation transformations before developing a native backend.

The input weight layout is `[K, N]`. Four contiguous input channels share a
`4 x 4` receiving-weight Gram matrix. At fixed reconstructed values and scales,
the selector compares the six possible two-of-four masks. Policies include
magnitude, receiving-row norm, quantization-aware diagonal gain and a signed
Gram score. Exact zero weights are handled by the coherence bound.

The quantizer reconstructs symmetric INT4/INT8 with per-last-axis-group maximum
scaling, round-to-nearest ties-to-even and signed range `[-7, 7]` / `[-127, 127]`.
Zero groups use scale one. Finite inputs are required. Values and metadata are
FP32; the output is reconstructed FP32, not packed low-bit execution. Format
switching, sparse instructions and matrix-tile scheduling are not implemented.

The gated policy uses the diagonal score when normalized off-diagonal row sum
`rho <= delta / (2 + delta)`. Under exact metadata, its loss for this quartet
and fixed candidate set is within factor `1 + delta` of the full local optimum.
It computes both paths in this reference, so timing it does not establish the
cost of selective hardware. Group losses omit cross-group interactions and
do not certify whole-product or model quality. Weight-derived metadata requires
preparation and storage; include both when assessing a deployed algorithm.

Run `python -m pytest tests/test_selection_reference.py`. The tests compare
against direct output products for all six masks, not the scalar expansion.

## Native CUDA mask evaluator

The optional Torch library also exports
`torch.ops.tensor_inference_engine.select_2of4(x, reconstructed, packed_gram,
diagonal=False)`. Inputs are contiguous FP32 matrices `[M, K]`; `K` is divisible
by four. Pack the upper triangle of each Gram block, in row-major order, into
`[K/4, 10]`. The result contains reconstructed FP32 values and uint8 pair indices
`[M, K/4]`, ordered `(01, 02, 03, 12, 13, 23)`. A native independent C++/CUDA
entry point is declared in `include/tensor_inference_engine/selection.hpp`.

The evaluator issues one CUDA kernel with a scalar thread per quartet. It
supports full or diagonal scores. Quantizer construction, metadata preparation,
format selection, sparse packing and GEMM are separate work. This is a GPU
implementation, not a synthesized autonomous hardware controller. Near ties
can differ from the Torch reference due to FP32 FMA; comparisons test objective
values. Finite inputs are required; no NaN/Inf policy is supplied.

After building the Torch adapter, run
`python -m pytest tests/test_selection_cuda.py`. Checks include stream ordering,
graph replay, operator registration, empty inputs and objective agreement.

## Fused current-block preparation

`torch.ops.tensor_inference_engine.prepare_activations(x, gram=None, bits=4,
policy=0)` accepts contiguous BF16 `[M,K]` with `K` divisible by32. It computes
the current maximum of each32-value group, reconstructs symmetric INT4/INT8,
casts the reconstruction to BF16 and optionally chooses a2:4 mask, in one
kernel. Policies0/1/2 mean dense, diagonal and full Gram selection. Sparse
policies require the same packed Gram metadata. Dense pair indices are255.

The scale uses multiplication by the rounded FP32 reciprocal of7/127,
matching Torch tensor/scalar division. This detail affects rounding ties;
the fused quantizer is tested against the reference before timing. The kernel
reads original activations once; it retains a32-value group within a warp until
its current maximum is known. No scale prediction or second global input read
is required. This is one producer-buffered baseline for speculative scaling.

Output values are BF16 reconstructions, with uint8 indices; they are not packed
integer data. There is no multi-mode tile controller or sparse GEMM here.
GPU fusion establishes implementation feasibility and latency, not custom-chip
area, energy, or the cost of buffering at a different producer boundary.

## Additional dense numerical baselines

`quantize_affine` reconstructs unsigned INT4/INT8 with per-group min/max
including zero and an integer zero point. `quantize_e2m1` reconstructs signed
E2M1 FP4 levels with nearest-even code rounding and dynamic maximum scaling.
Its scales stay FP32; it is not a full NVFP4/MXFP4 recipe. Both references
require finite inputs, preserve all tensor positions and return FP32 values.
Their scale/offset arithmetic and execution costs differ from symmetric INT.
