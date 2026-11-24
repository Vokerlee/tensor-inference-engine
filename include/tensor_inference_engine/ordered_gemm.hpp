#pragma once

#include "tensor_inference_engine/gemm.hpp"

namespace tensor_inference_engine {

// FP32 output of independently reset primitive_k dot products, merged in K
// order. Each primitive uses one sequential RN FMA chain, or four independent
// interleaved chains followed by a fixed RN binary reduction. The latter is a
// different declared arithmetic contract, not an equivalent optimization.
// Optional indices must be sorted, unique flattened output positions. Other
// output entries are left untouched. Invalid indices are skipped and counted.
// invalid_count points to a caller-zeroed device uint64; all storage is borrowed.
void ordered_gemm(Bf16Matrix a, Bf16Matrix b, GemmOutput out, unsigned primitive_k,
    unsigned accumulators, const std::int64_t* indices, std::int64_t count,
    std::uint64_t* invalid_count, cudaStream_t stream = nullptr);

} // namespace tensor_inference_engine
