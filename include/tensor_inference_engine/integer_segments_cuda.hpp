#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensor_inference_engine {

// Row statistics [rows, groups], window statistics [windows, groups].
// Magnitudes/L1 norms are nonnegative int32. Exponents describe exact powers
// of two; inactive weight windows may use zero exponent endpoints.
// Optional row_l1/window_max must either both be present or both be null.
// Plans [rows, windows, ceil(groups/groups_per_slice)] use byte end/oversize
// masks and a validity flag. No operand products or expanded bound tensors.
void plan_dot_integer_segments(const std::int32_t* row_max, const std::int32_t* row_exponent,
    const std::int32_t* window_l1, const std::int32_t* window_low, const std::int32_t* window_high,
    const std::int32_t* row_l1, const std::int32_t* window_max, std::uint8_t* ends,
    std::uint8_t* oversized, bool* valid, std::int64_t rows, std::int64_t windows,
    std::int64_t groups, int groups_per_slice, unsigned bits, cudaStream_t stream);

} // namespace tensor_inference_engine
