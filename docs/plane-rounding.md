# Bounded-state plane rounding

The C++ and optional Torch operators quantize a contiguous FP32 tensor
`[planes, rows, channels]`. Each plane observes the current maximum in groups
of32,128 or256 channels and selects a symmetric power-of-two INT4/8 grid.
A zero group uses scale1. The interface supports1–36 planes and empty rows.
Prepared metadata is shared by all rows; it contains no model-specific objects.

`plane_rounding` accepts FP32 factors `[channels, planes, rank]` with rank1 or4,
and nonnegative FP32 diagonal terms `[channels, planes]`. It greedily chooses
adjacent floor/ceil levels, then compares the completed surrogate with nearest
rounding. This terminal selection requires retained codes or replay before
emission; it is not a causal per-plane output protocol.

`plane_shift_rounding` uses one scalar error state and INT16 metadata
`[channels, planes]`. Its low byte is a signed direction code: zero means zero,
and a nonzero code with magnitude k represents a signed coefficient
`2**(-(k-1))`. The high byte is a nonnegative diagonal code with the same
power-of-two magnitude convention. Nonzero magnitudes must lie in1–127.
For direction u and total diagonal h, the caller must supply `h >= u*u`;
invalid encodings or a negative implied diagonal residual are rejected by the
deferred audit counter.

The causal variant chooses an adjacent level using current errors eL,eH and
state v through `2*u*v + (eL+eH)*h`. Positive values choose the lower level,
negative values the upper level, and ties retain nearest. It updates
`v += u*e` and immediately emits that plane. Coefficient products are exponent
shifts; no squared cost or terminal fallback is evaluated. This greedy rule
has no guarantee of reducing the complete sequence's error against nearest.

Both operators return reconstructed FP32 values, INT32 group exponents, a
UINT8 row/channel flag and an INT64 invalid-channel counter. The floating
variant's flag means completed proxy selection; the causal variant's flag
means at least one emitted code differs from nearest. Explicit RN FP32
operations define addition/multiplication order; logarithmic scale observation
uses the CUDA math implementation.

The default Torch call checks invalid arithmetic before returning. With
`check_finite=False`, callers must inspect the counter before consuming values.
No allocation or synchronization occurs in the raw C++ interface. Torch handles
allocation, device/stream guards and Meta shapes; inference with gradients is
rejected. Nondefault CUDA streams are supported.

One CUDA kernel is an experimental execution implementation. Its duration does
not specify the latency, ports, bandwidth or energy of another processor.
