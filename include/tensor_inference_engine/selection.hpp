#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensor_inference_engine {

// Contiguous FP32 x/q/output [rows, channels], channels divisible by four.
// gram stores the ten upper-triangle entries per quartet in row-major order.
// pairs [rows, channels/4] uses (01, 02, 03, 12, 13, 23). Inputs must be finite.
void select_2of4(const float* x, const float* q, const float* gram, float* output,
    std::uint8_t* pairs, std::int64_t rows, std::int64_t channels, bool diagonal,
    cudaStream_t stream);

} // namespace tensor_inference_engine
