#pragma once

#include <math_constants.h>

namespace tensor_inference_engine::detail {

__device__ inline int first(int pair)
{
    return pair < 3 ? 0 : (pair < 5 ? 1 : 2);
}

__device__ inline int second(int pair)
{
    return pair == 0 ? 1 : ((pair == 1 || pair == 3) ? 2 : 3);
}

template<bool Diagonal>
__device__ int choose_pair(const float (&values)[4], const float (&quantized)[4], const float* g)
{
    constexpr int upper[4][4] = {
        {0, 1, 2, 3},
        {1, 4, 5, 6},
        {2, 5, 7, 8},
        {3, 6, 8, 9}
    };
    float terms[4];
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
        terms[i] = fmaf(2 * quantized[i], response, quantized[i] * quantized[i] * g[upper[i][i]]);
    }
    float best = CUDART_INF_F;
    int chosen = 0;
#pragma unroll
    for (int pair = 0; pair < 6; ++pair) {
        const int i = first(pair), j = second(pair);
        float score = terms[i] + terms[j];
        if constexpr (!Diagonal)
            score = fmaf(2 * quantized[i] * quantized[j], g[upper[i][j]], score);
        if (score < best) {
            best = score;
            chosen = pair;
        }
    }
    return chosen;
}

} // namespace tensor_inference_engine::detail
