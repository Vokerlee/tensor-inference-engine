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
