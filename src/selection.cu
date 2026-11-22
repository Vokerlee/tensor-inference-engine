#include "tensor_inference_engine/selection.hpp"

#include <math_constants.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

template<bool Diagonal>
__global__ void select_kernel(const float* x, const float* q, const float* gram, float* output,
    std::uint8_t* pairs, std::int64_t count, std::int64_t groups)
{
    constexpr int upper[4][4] = {
        {0, 1, 2, 3},
        {1, 4, 5, 6},
        {2, 5, 7, 8},
        {3, 6, 8, 9}
    };
    constexpr int first[6] = {0, 0, 0, 1, 1, 2};
    constexpr int second[6] = {1, 2, 3, 2, 3, 3};
    for (std::int64_t index = blockIdx.x * std::int64_t(blockDim.x) + threadIdx.x; index < count;
        index += gridDim.x * std::int64_t(blockDim.x)) {
        const auto base = index * 4;
        const auto* g = gram + (index % groups) * 10;
        float values[4], quantized[4], terms[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[i] = x[base + i];
            quantized[i] = q[base + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float response = 0;
            if constexpr (Diagonal) {
                response = -values[i] * g[upper[i][i]];
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    response = fmaf(-values[j], g[upper[i][j]], response);
            }
            terms[i] =
                fmaf(2 * quantized[i], response, quantized[i] * quantized[i] * g[upper[i][i]]);
        }
        float best = CUDART_INF_F;
        int chosen = 0;
#pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int i = first[pair], j = second[pair];
            float score = terms[i] + terms[j];
            if constexpr (!Diagonal)
                score = fmaf(2 * quantized[i] * quantized[j], g[upper[i][j]], score);
            if (score < best) {
                best = score;
                chosen = pair;
            }
        }
        pairs[index] = static_cast<std::uint8_t>(chosen);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            output[base + i] = (i == first[chosen] || i == second[chosen]) ? quantized[i] : 0;
    }
}

} // namespace

void select_2of4(const float* x, const float* q, const float* gram, float* output,
    std::uint8_t* pairs, std::int64_t rows, std::int64_t channels, bool diagonal,
    cudaStream_t stream)
{
    if (rows < 0 || channels < 0 || channels % 4)
        throw std::invalid_argument(
            "selection requires nonnegative rows and channels divisible by four"
        );
    if (!rows || !channels)
        return;
    constexpr int threads = 128;
    const auto groups = channels / 4;
    const auto count = rows * groups;
    const auto blocks =
        static_cast<int>(std::min<std::int64_t>((count + threads - 1) / threads, 65535));
    if (diagonal)
        select_kernel<true>
            <<<blocks, threads, 0, stream>>>(x, q, gram, output, pairs, count, groups);
    else
        select_kernel<false>
            <<<blocks, threads, 0, stream>>>(x, q, gram, output, pairs, count, groups);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("selection kernel: ") + cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
