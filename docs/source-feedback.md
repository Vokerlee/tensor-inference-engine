# Source-group quantization with row-local feedback

The C++ `source_feedback` primitive accepts contiguous BF16
`[streams,width,channels]`, with channels divisible by32, and one FP32 feedback
coefficient per channel. Each current ORIGINAL group of32 values chooses a
power-of-two symmetric INT4/INT8 scale. Width is scanned causally; each stream
starts with zero carry. Coefficients must be finite in[-1,1].

For each element, form `incoming = source + coefficient * carry`, quantize with
round-to-nearest-even and clipping, then update `carry = incoming - quantized`.
Optional zero preservation emits zero and clears carry for exact source zeros.
The CUDA kernel uses explicit FP32 multiply/add/subtract rounding, without
fusing the multiply/add. Group scales never depend on future elements or on
feedback-adjusted values.

The operator returns BF16 reconstructed values, terminal FP32 carry per
stream/channel, int32 exponents per source group, and three int64 counters:
clipping, introduced nonzeros before BF16 output conversion, and invalid
input/arithmetic events. This is a numerical reconstruction operator, not a
packed INT4/INT8 matrix kernel. Model-specific coefficient selection and tensor
layout conversion belong to callers.

The native C++ API allocates nothing and uses the caller's CUDA stream. The
caller initializes counters to zero and must reject results with invalid events.
The Torch adapter checks those events synchronously by default. Passing
`check_finite=False` defers validation and permits CUDA graph capture; callers
must audit the returned invalid counter before accepting numerical results.

```python
output, terminal, exponents, counts = torch.ops.tensor_inference_engine.source_feedback(
    values, coefficient, bits=4, preserve_zeros=True
)
```

Shape/dtype/device and inference-only constraints are also available to the
Meta implementation. CPU-reference tests cover mixed coefficients, original
scales, clipping, exact zeros, stream resets, empty dimensions, invalid values
and repeated graph execution. GPU execution cost does not estimate a dedicated
quantizer's latency, state implementation or energy.

An optional `fixed_exponents` INT32 tensor with shape
`[streams,width,channels/32]` overrides current-group scale selection. Inputs,
feedback recurrence, row resets and clipping remain live. This supports caller
supplied POT scales and exact replay tests; it does not certify that those
scales are suitable for changed inputs. Invalid scales/arithmetic are exposed
through the existing validation counter. Shape/dtype/device checks also apply
to Meta allocation. Output exponents record the scales actually used.
