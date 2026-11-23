#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensor_inference_engine {

// Contiguous BF16 [streams, width, channels], channels divisible by 32.
// Current original groups of 32 channels select a power-of-two INT4/INT8 scale.
// A row-local FP32 carry follows u=x+coefficient*carry, q=Q(u), carry=u-q.
// Output q is converted once to BF16; terminal carry remains FP32. Exact-zero
// preservation additionally emits zero and clears carry at a zero source.
// Exponents are int32 [streams,width,channels/32]; terminal is [streams,channels].
// Optional fixed_exponents supplies external POT scales with the same shape.
// This overrides scale selection only; original values and feedback stay live.
// Counts must be initialized to zero: clipping, introduced nonzeros before BF16
// conversion, and invalid input/arithmetic events. Nonzero invalid count means
// numerical outputs must not be consumed. Coefficients must be finite in [-1,1].
// Pointers belong to the caller's current CUDA device, with no aliasing and
// lifetime through stream completion. No allocation or device sync is performed.
void source_feedback(const std::uint16_t* source, const float* coefficient, std::uint16_t* output,
    float* terminal, std::int32_t* exponents, std::uint64_t* counts, std::int64_t streams,
    std::int64_t width, std::int64_t channels, unsigned bits, bool preserve_zeros,
    cudaStream_t stream = nullptr, const std::int32_t* fixed_exponents = nullptr);

} // namespace tensor_inference_engine
