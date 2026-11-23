# Bounded integer segment planning

`integer_segments.hpp` provides a C++/CUDA-compatible scalar planner for at most
eight consecutive integer contribution bounds. It has no model, tensor-layout
or matrix-unit geometry dependency.

Each group supplies an unsigned magnitude bound, a minimum admissible exponent
and a maximum exponent. A segment uses the minimum participating exponent as
its common anchor and conservatively shifts all magnitude bounds into that
domain. It extends while their sum fits the positive limit of a signed integer
with the requested width. Zero groups do not change the anchor. Widths2..63
are supported; active exponent endpoints must be within the documented int32
range used by the implementation.

The result contains an end-bit mask, an oversized-group mask and a validity
flag. A caller can send oversized single groups to its fallback arithmetic.
Invalid arguments or active exponent intervals return an invalid plan. The
empty sequence has an empty valid plan. Shift/add guards saturate before any
unsigned overflow or excessive shift can occur.

The planner minimizes the number of contiguous segments under these bounds
and a single-active-group fallback. It does not minimize physical energy or
guarantee equality to another floating-point summation order. Callers own
metadata correctness, representation changes and numerical validation.

`integer_segments_native` checks emitted partitions against exhaustive choices,
including zero groups, exponent gaps, oversized magnitudes and invalid ranges.

## Fused CUDA metadata interface

`integer_segments_cuda.hpp` combines row maximum/exponent metadata with window
L1/exponent extrema and computes conservative product bounds in registers.
Optional row-L1/window-maximum metadata supplies the tighter alternative bound.
It plans each output window and group slice without materializing expanded
per-pair bound tensors or accessing the original matrix operands.

The optional Torch operator `integer_segment_plan` accepts contiguous int32
statistics `[rows, groups]` and `[windows, groups]`. Its three outputs have shape
`[rows, windows, ceil(groups/groups_per_slice)]`: byte end masks, byte oversized
masks and Boolean validity. CUDA slices contain1..8 groups with widths2..48.
Neither group size in elements nor window size in output columns is fixed by
this interface.

Caller-supplied statistics must bound the represented operands and follow their
layout. Detected negative/inconsistent norms or invalid active exponent ranges
return an invalid all-single-group fallback plan. This is not a replacement
for correct metadata ownership. The kernel honors the caller's CUDA stream;
the adapter also supports Meta tensors and CUDA graph capture.

`integer_segments_cuda` checks fused indexing, tail slices and both bound modes
against the scalar planner. These checks do not measure the cost of generating
source statistics or establish a hardware energy benefit.
