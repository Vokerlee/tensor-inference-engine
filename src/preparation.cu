#include "selection.cuh"
#include "tensor_inference_engine/selection.hpp"

#include <algorithm>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

template<int Policy>
__global__ void prepare_kernel(const __nv_bfloat16* x, const float* gram, __nv_bfloat16* output,
    std::uint8_t* pairs, std::int64_t count, std::int64_t channels, float limit)
{
    const int lane = threadIdx.x % 32;
    for (std::int64_t index = blockIdx.x * std::int64_t(blockDim.x) + threadIdx.x; index < count;
        index += gridDim.x * std::int64_t(blockDim.x)) {
        const float value = __bfloat162float(x[index]);
        float maximum = fabsf(value);
#pragma unroll
        for (int distance = 16; distance > 0; distance /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, distance));
        // Torch tensor/scalar division multiplies by the rounded reciprocal.
        const float scale = maximum > 0 ? __fmul_rn(maximum, __frcp_rn(limit)) : 1.0f;
        const float integer = fminf(limit, fmaxf(-limit, rintf(value / scale)));
        const float quantized = __bfloat162float(__float2bfloat16_rn(integer * scale));
        if constexpr (Policy == 0) {
            output[index] = __float2bfloat16_rn(quantized);
            if (lane % 4 == 0)
                pairs[index / 4] = 255;
        } else {
            float values[4], codes[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                values[i] = __shfl_sync(0xffffffff, value, (lane / 4) * 4 + i);
                codes[i] = __shfl_sync(0xffffffff, quantized, (lane / 4) * 4 + i);
            }
            if (lane % 4 == 0) {
                const auto* g = gram + ((index % channels) / 4) * 10;
                const int chosen = detail::choose_pair<Policy == 1>(values, codes, g);
                pairs[index / 4] = static_cast<std::uint8_t>(chosen);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    output[index + i] = __float2bfloat16_rn(
                        i == detail::first(chosen) || i == detail::second(chosen) ? codes[i] : 0
                    );
            }
        }
    }
}

} // namespace

void prepare_activations(const std::uint16_t* x, const float* gram, std::uint16_t* output,
    std::uint8_t* pairs, std::int64_t rows, std::int64_t channels, int bits, int policy,
    cudaStream_t stream)
{
    if (rows < 0 || channels < 0 || channels % 32 || (bits != 4 && bits != 8) || policy < 0 ||
        policy > 2)
        throw std::invalid_argument("preparation needs K divisible by32,4/8 bits,policy0/1/2");
    if (!rows || !channels)
        return;
    const auto count = rows * channels;
    constexpr int threads = 128;
    const int blocks =
        static_cast<int>(std::min<std::int64_t>((count + threads - 1) / threads, 65535));
    const auto* input = reinterpret_cast<const __nv_bfloat16*>(x);
    auto* result = reinterpret_cast<__nv_bfloat16*>(output);
    const float limit = bits == 4 ? 7.0f : 127.0f;
    if (policy == 0)
        prepare_kernel<0>
            <<<blocks, threads, 0, stream>>>(input, gram, result, pairs, count, channels, limit);
    else if (policy == 1)
        prepare_kernel<1>
            <<<blocks, threads, 0, stream>>>(input, gram, result, pairs, count, channels, limit);
    else
        prepare_kernel<2>
            <<<blocks, threads, 0, stream>>>(input, gram, result, pairs, count, channels, limit);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("preparation kernel: ") + cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
