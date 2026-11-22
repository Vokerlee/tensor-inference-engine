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

// BF16 inputs and reconstructions, scale group32, symmetric INT4/INT8.
// policy0: dense (gram may be null, pairs255);1: diagonal2:4;2: full2:4.
void prepare_activations(const std::uint16_t* x, const float* gram, std::uint16_t* output,
    std::uint8_t* pairs, std::int64_t rows, std::int64_t channels, int bits, int policy,
    cudaStream_t stream);

} // namespace tensor_inference_engine
