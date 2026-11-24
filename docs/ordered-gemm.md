# Explicit FP32 accumulation orders

`ordered_gemm` computes BF16 matrix products with FP32 outputs under a declared
operation graph. Each `primitive_k` reduction slice starts at positive zero;
slices are merged in increasing K order with round-to-nearest FP32 addition.
One accumulator performs sequential RN FMA. Four accumulators interleave input
positions and combine the four chains with a fixed balanced RN addition tree.
These are different arithmetic contracts and can produce different results.
Neither is asserted to match an unspecified accelerator or Tensor Core.

The C++ API borrows operand/output/index/counter storage and the caller's stream.
The Torch adapter accepts contiguous `[M,K]`, `[K,N]` BF16 operands and returns
`(output_fp32, invalid_indices_int64)`. This operator never rounds its matrix
output to BF16. Empty K has a zero result; empty M/N are defined.

Full outputs share4x64 operand tiles with32-element loads; each output keeps
the declared per-primitive FMA order. Sparse outputs use compact indexed
threads. Dense tiling and indexed execution are tested against one another
across row, column and reduction tails.

Optional sorted, unique INT64 flattened output indices compute only selected
elements, preserving the supplied initial FP32 matrix elsewhere. The adapter
clones initial storage. Empty selection preserves every entry. Native bounds
and ordering checks skip invalid indices and count them; default Torch behavior
rejects any such event. With `check_indices=False`, callers must audit the
returned counter before accepting the result. This deferred mode is for trusted
selection pipelines, not permission to accept invalid results.

```python
reference, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 1)
parallel_chains, invalid = torch.ops.tensor_inference_engine.ordered_gemm(a, b, 128, 4)
repaired, invalid = torch.ops.tensor_inference_engine.ordered_gemm(
    a, b, 128, 1, selected_flat_indices, initial_fp32)
```

This is an inference-only numerical primitive. Model wrappers, acceptance/error
criteria and scheduling experiments belong to the caller. Timing this kernel
measures its software implementation, not the modeled device's latency or energy.
