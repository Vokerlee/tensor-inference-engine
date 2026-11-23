#include "tensor_inference_engine/source_feedback.hpp"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

__global__ void feedback_kernel(const __nv_bfloat16* source, const float* coefficient,
    __nv_bfloat16* output, float* terminal, std::int32_t* exponents, unsigned long long* counts,
    std::int64_t streams, std::int64_t width, std::int64_t channels, float limit,
    bool preserve_zeros)
{
    constexpr unsigned mask = 0xffffffffu;
    const auto lane = threadIdx.x % 32;
    const auto groups = channels / 32;
    const auto stride = std::int64_t(gridDim.x) * blockDim.x / 32;
    for (std::int64_t work = std::int64_t(blockIdx.x) * blockDim.x / 32 + threadIdx.x / 32;
        work < streams * groups; work += stride) {
        const auto row = work / groups;
        const auto group = work % groups;
        const auto channel = group * 32 + lane;
        const float supplied = coefficient[channel];
        const bool bad_coefficient = !isfinite(supplied) || fabsf(supplied) > 1;
        const float alpha = bad_coefficient ? 0 : supplied;
        float carry = 0;
        unsigned long long clipped = 0;
        unsigned long long introduced = 0;
        unsigned long long invalid = bad_coefficient;
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = (row * width + column) * channels + channel;
            const float loaded = __bfloat162float(source[index]);
            const bool bad_source = !isfinite(loaded);
            invalid += bad_source;
            const float original = bad_source ? 0 : loaded;
            float maximum = fabsf(original);
            for (int offset = 16; offset; offset /= 2)
                maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, offset));
            maximum = __shfl_sync(mask, maximum, 0);
            int exponent = 0;
            float scale = 1;
            bool bad_scale = false;
            if (lane == 0 && maximum > 0) {
                const float logarithm = log2f(__fdiv_rn(maximum, limit));
                bad_scale = !isfinite(logarithm);
                if (!bad_scale) {
                    exponent = static_cast<int>(ceilf(logarithm));
                    scale = ldexpf(1.0f, exponent);
                    bad_scale = !isfinite(scale) || scale <= 0;
                }
            }
            exponent = __shfl_sync(mask, exponent, 0);
            scale = __shfl_sync(mask, scale, 0);
            bad_scale = __shfl_sync(mask, int(bad_scale), 0);
            invalid += bad_scale;
            if (bad_scale)
                scale = 1;
            if (lane == 0)
                exponents[(row * width + column) * groups + group] = exponent;
            const float incoming = __fadd_rn(original, __fmul_rn(alpha, carry));
            const float unbounded = nearbyintf(__fdiv_rn(incoming, scale));
            clipped += fabsf(unbounded) > limit;
            float current = __fmul_rn(fminf(limit, fmaxf(-limit, unbounded)), scale);
            carry = __fsub_rn(incoming, current);
            if (preserve_zeros && original == 0) {
                current = 0;
                carry = 0;
            }
            const auto converted = __float2bfloat16_rn(current);
            invalid +=
                !isfinite(incoming) || !isfinite(carry) || !isfinite(__bfloat162float(converted));
            introduced += original == 0 && current != 0;
            output[index] = converted;
        }
        terminal[row * channels + channel] = carry;
        for (int offset = 16; offset; offset /= 2) {
            clipped += __shfl_down_sync(mask, clipped, offset);
            introduced += __shfl_down_sync(mask, introduced, offset);
            invalid += __shfl_down_sync(mask, invalid, offset);
        }
        if (lane == 0) {
            atomicAdd(counts, clipped);
            atomicAdd(counts + 1, introduced);
            atomicAdd(counts + 2, invalid);
        }
    }
}

} // namespace

void source_feedback(const std::uint16_t* source, const float* coefficient, std::uint16_t* output,
    float* terminal, std::int32_t* exponents, std::uint64_t* counts, std::int64_t streams,
    std::int64_t width, std::int64_t channels, unsigned bits, bool preserve_zeros,
    cudaStream_t stream)
{
    if (streams < 0 || width < 0 || channels <= 0 || channels % 32 || (bits != 4 && bits != 8))
        throw std::invalid_argument("feedback requires aligned nonnegative shapes and bits4/8");
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    if (streams > maximum / channels || (streams && width > maximum / streams / channels))
        throw std::invalid_argument("feedback shape product overflows int64");
    if (!streams)
        return;
    if (!coefficient || !terminal || !counts || (width && (!source || !output || !exponents)))
        throw std::invalid_argument("feedback requires valid nonaliasing device pointers");
    constexpr int threads = 128;
    const auto warps = streams * (channels / 32);
    const auto blocks = static_cast<int>(std::min<std::int64_t>((warps + 3) / 4, 65535));
    feedback_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(source),
        coefficient, reinterpret_cast<__nv_bfloat16*>(output), terminal, exponents,
        reinterpret_cast<unsigned long long*>(counts), streams, width, channels,
        float((1u << (bits - 1)) - 1), preserve_zeros);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("feedback kernel: ") + cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
