# Directed norm bounds and explicit BF16 observations

`bf16_norms(values, columns=False)` takes a contiguous BF16 matrix and returns
FP32 `[L1,Linf,L2,bad]` bounds per row, or per column when requested. Positive
sums, products and square roots round upward explicitly. Nonfinite inputs and
BF16 subnormals set `bad`; they are excluded from observation acceptance.
Overflow of a norm is conservative. No fitted activation statistics are used.

`bf16_observe(product, source_norms, weight_norms, coefficient, underflow,
bias=None)` takes a FP32 product and norm bounds for its actual immutable
operands. It uses the minimum of three norm-product upper bounds for
`S=sum(abs(a_k*b_k))`. The caller must establish a valid difference bound
`coefficient*S + underflow` for its two declared arithmetic graphs. Shape/dtype
checks do not prove that supplied metadata correspond to the operands.

The result is `(observed_bf16, accepted_bool, error_fp32)`. Optional FP32 bias
is added with RN arithmetic and its extra rounding error is included. A result
is accepted only when a directed interval lies strictly within one BF16
rounding cell, or when zero operand products determine the observation exactly.
Ties, unsupported extreme/subnormal output cells, bad source rows/columns and
unbounded intermediate ranges remain unaccepted. Neighboring BF16 values and
explicit CUDA rounding intrinsics preserve the intended precision boundary.

This operation is an explicitly selected observation of an already computed
FP32 matrix. It does not change the GEMM output contract or certify equality
for FP32 consumers, adaptive floating-point state or unmodeled vector operations.
Acceptance is conditional on valid caller coefficients and immutable operand
metadata. No device-specific arithmetic model or model architecture is assumed.
