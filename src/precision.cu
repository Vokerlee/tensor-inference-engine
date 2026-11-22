#include "tensor_inference_engine/selection.hpp"

#include <cmath>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

__device__ float reconstruct(float value, float maximum, int bits, bool fp4)
{
    const float limit = bits == 8 ? 127.0f : (fp4 ? 6.0f : 7.0f);
    const float scale = maximum > 0 ? __fmul_rn(maximum, __frcp_rn(limit)) : 1.0f;
    float code;
    if (bits == 4 && fp4) {
        constexpr float boundaries[7] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        const float magnitude = fabsf(value) / scale;
        int index = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i) {
            index += magnitude > boundaries[i];
            index += magnitude == boundaries[i] && i % 2 == 1;
        }
        const float level = index < 4
            ? .5f * index
            : (index == 4 ? 2.0f : (index == 5 ? 3.0f : (index == 6 ? 4.0f : 6.0f)));
        code = copysignf(level, value);
    } else {
        code = fminf(limit, fmaxf(-limit, rintf(value / scale)));
    }
    return __bfloat162float(__float2bfloat16_rn(code * scale));
}

template<bool Diagonal>
__device__ float quadratic(const float (&values)[4], const float* gram)
{
    constexpr int diagonal[4] = {0, 4, 7, 9};
    float result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        result = fmaf(values[i] * values[i], gram[diagonal[i]], result);
    if constexpr (!Diagonal) {
        constexpr int first[6] = {0, 0, 0, 1, 1, 2};
        constexpr int second[6] = {1, 2, 3, 2, 3, 3};
        constexpr int upper[6] = {1, 2, 3, 5, 6, 8};
#pragma unroll
        for (int pair = 0; pair < 6; ++pair)
            result =
                fmaf(2 * values[first[pair]] * values[second[pair]], gram[upper[pair]], result);
    }
    return fmaxf(result, 0);
}

template<bool LowDiagonal, bool HighDiagonal, bool ReferenceDiagonal>
__global__ void precision_kernel(const std::uint16_t* x, const float* gram, std::uint16_t* output,
    std::uint8_t* modes, std::int64_t rows, std::int64_t channels, float tolerance, bool fp4)
{
    __shared__ std::uint16_t raw[128 * 4 * 34];
    __shared__ float maxima[128 * 4];
    __shared__ float geometry[32 * 10];
    __shared__ float reduction[3][256];
    __shared__ int mode;
    auto* values = reinterpret_cast<__nv_bfloat16*>(raw);
    const auto row_base = std::int64_t(blockIdx.y) * 128;
    const auto column_base = std::int64_t(blockIdx.x) * 128;
    for (int index = threadIdx.x; index < 128 * 128; index += 256) {
        const auto row = row_base + index / 128, column = column_base + index % 128;
        raw[(index / 32) * 34 + index % 32] =
            row < rows && column < channels ? x[row * channels + column] : 0;
    }
    for (int index = threadIdx.x; index < 32 * 10; index += 256) {
        const auto group = column_base / 4 + index / 10;
        geometry[index] = group < channels / 4 ? gram[group * 10 + index % 10] : 0;
    }
    __syncthreads();
    float low_error = 0, high_error = 0, reference = 0;
    for (int group = threadIdx.x; group < 128 * 4; group += 256) {
        const int offset = group * 34;
        float maximum = 0;
#pragma unroll
        for (int i = 0; i < 32; ++i)
            maximum = fmaxf(maximum, fabsf(__bfloat162float(values[offset + i])));
        maxima[group] = maximum;
#pragma unroll
        for (int quartet = 0; quartet < 8; ++quartet) {
            float original[4], low[4], high[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                original[i] = __bfloat162float(values[offset + quartet * 4 + i]);
                low[i] = reconstruct(original[i], maximum, 4, fp4) - original[i];
                high[i] = reconstruct(original[i], maximum, 8, false) - original[i];
            }
            const auto* g = geometry + ((group % 4) * 8 + quartet) * 10;
            low_error += quadratic<LowDiagonal>(low, g);
            high_error += quadratic<HighDiagonal>(high, g);
            reference += quadratic<ReferenceDiagonal>(original, g);
        }
    }
    reduction[0][threadIdx.x] = low_error;
    reduction[1][threadIdx.x] = high_error;
    reduction[2][threadIdx.x] = reference;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                reduction[i][threadIdx.x] += reduction[i][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float denominator = fmaxf(reduction[2][0], 1e-30f);
        mode = reduction[0][0] / denominator <= tolerance
            ? 0
            : (reduction[1][0] / denominator <= tolerance ? 1 : 2);
        modes[blockIdx.y * gridDim.x + blockIdx.x] = static_cast<std::uint8_t>(mode);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < 128 * 128; index += 256) {
        const auto row = row_base + index / 128, column = column_base + index % 128;
        if (row < rows && column < channels) {
            if (mode == 2) {
                output[row * channels + column] = raw[(index / 32) * 34 + index % 32];
            } else {
                const auto quantized = __float2bfloat16_rn(
                    reconstruct(__bfloat162float(values[(index / 32) * 34 + index % 32]),
                        maxima[index / 32], mode == 0 ? 4 : 8, fp4)
                );
                reinterpret_cast<__nv_bfloat16*>(output)[row * channels + column] = quantized;
            }
        }
    }
}

} // namespace

void adaptive_precision(const std::uint16_t* x, const float* gram, std::uint16_t* output,
    std::uint8_t* modes, std::int64_t rows, std::int64_t channels, float tolerance, bool fp4,
    PrecisionGeometry geometry, cudaStream_t stream)
{
    if (rows < 0 || channels < 0 || channels % 32 || !std::isfinite(tolerance) || tolerance <= 0)
        throw std::invalid_argument(
            "precision selection needs K divisible by32 and positive finite tolerance"
        );
    if (!rows || !channels)
        return;
    const dim3 grid((channels + 127) / 128, (rows + 127) / 128);
    switch (geometry) {
    case PrecisionGeometry::full:
        precision_kernel<false, false, false>
            <<<grid, 256, 0, stream>>>(x, gram, output, modes, rows, channels, tolerance, fp4);
        break;
    case PrecisionGeometry::diagonal:
        precision_kernel<true, true, true>
            <<<grid, 256, 0, stream>>>(x, gram, output, modes, rows, channels, tolerance, fp4);
        break;
    case PrecisionGeometry::low_error:
        precision_kernel<false, true, true>
            <<<grid, 256, 0, stream>>>(x, gram, output, modes, rows, channels, tolerance, fp4);
        break;
    case PrecisionGeometry::signal:
        precision_kernel<true, true, false>
            <<<grid, 256, 0, stream>>>(x, gram, output, modes, rows, channels, tolerance, fp4);
        break;
    default:
        throw std::invalid_argument("unsupported precision geometry");
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("precision kernel: ") + cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
