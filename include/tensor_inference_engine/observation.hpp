#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensor_inference_engine {

// Row/column [L1,Linf,L2,bad] bounds, with directed-up FP32 arithmetic.
// Nonfinite and BF16-subnormal sources set bad. Output has four floats/row.
void bf16_norms(const std::uint16_t* values, std::int64_t rows, std::int64_t columns,
    bool along_columns, float* output, cudaStream_t stream = nullptr);

// Explicit BF16 observation of a FP32 product (plus optional FP32 bias).
// Norms must describe its actual immutable operands. Caller supplies valid
// nonnegative upper coefficients for error <= coefficient*S + underflow.
// Matrix output itself remains FP32; this is a separately chosen observation.
void bf16_observe(const float* product, const float* source_norms, const float* weight_norms,
    const float* bias, std::uint16_t* observed, bool* accepted, float* error, std::int64_t rows,
    std::int64_t columns, float coefficient, float underflow, cudaStream_t stream = nullptr);

} // namespace tensor_inference_engine
