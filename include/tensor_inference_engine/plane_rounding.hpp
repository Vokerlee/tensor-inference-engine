#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensor_inference_engine {

// FP32 source [planes,rows,channels], factor [channels,planes,rank] and
// nonnegative diagonal [channels,planes]. planes is in [1,36], rank is 1 or 4,
// group is 32/128/256 and divides channels. Metadata is shared by every row.
// Current group maxima select POT INT4/8 grids. Adjacent code decisions use a
// rank-sized error state; completed surrogate selection can retain nearest.
// Outputs: FP32 reconstructed source, INT32 [planes,rows,channels/group]
// exponents, UINT8 [rows,channels] selection, and one zero-initialized UINT64
// invalid-channel counter. Nonzero counter forbids consumption of the outputs.
// Arithmetic uses explicit FP32 RN add/multiply without implicit FMA contraction.
// Metadata preparation, allocation and device synchronization belong to caller.
// All pointers are nonaliasing on the current device, live until stream completion.
void plane_rounding(const float* source, const float* factor, const float* diagonal,
    float* reconstructed, std::int32_t* exponents, std::uint8_t* selected, std::uint64_t* invalid,
    std::int64_t planes, std::int64_t rows, std::int64_t channels, unsigned rank, unsigned bits,
    unsigned group, cudaStream_t stream = nullptr);

} // namespace tensor_inference_engine
