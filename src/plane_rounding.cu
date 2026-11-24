#include "tensor_inference_engine/plane_rounding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

__device__ float warp_maximum(float value)
{
    for (int offset = 16; offset; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

template<unsigned Rank>
__global__ void rounding_kernel(const float* source, const float* factor, const float* diagonal,
    float* output, std::int32_t* exponents, std::uint8_t* selected, unsigned long long* invalid,
    std::int64_t planes, std::int64_t rows, std::int64_t channels, float limit, unsigned group)
{
    __shared__ float maxima[8];
    __shared__ float current_scale;
    __shared__ int current_exponent;
    __shared__ int bad_scale;
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    const auto groups = channels / group;
    for (std::int64_t work = blockIdx.x; work < rows * groups; work += gridDim.x) {
        const auto row = work / groups;
        const auto channel = (work % groups) * group + threadIdx.x;
        float state[Rank] = {};
        float nearest_state[Rank] = {};
        std::int8_t candidate[36];
        float candidate_diagonal = 0;
        float nearest_diagonal = 0;
        bool valid = true;
        for (std::int64_t plane = 0; plane < planes; ++plane) {
            const auto index = (plane * rows + row) * channels + channel;
            const float loaded = source[index];
            valid = valid && isfinite(loaded);
            const float original = isfinite(loaded) ? loaded : 0;
            const float reduced = warp_maximum(fabsf(original));
            if (lane == 0)
                maxima[warp] = reduced;
            __syncthreads();
            if (warp == 0) {
                const float maximum = warp_maximum(lane < group / 32 ? maxima[lane] : 0);
                if (lane == 0) {
                    current_exponent = 0;
                    current_scale = 1;
                    bad_scale = 0;
                    if (maximum > 0) {
                        const float logarithm = log2f(__fdiv_rn(maximum, limit));
                        bad_scale = !isfinite(logarithm);
                        if (!bad_scale) {
                            current_exponent = static_cast<int>(ceilf(logarithm));
                            current_scale = ldexpf(1.0f, current_exponent);
                            bad_scale = !isfinite(current_scale) || current_scale <= 0;
                        }
                    }
                    if (bad_scale)
                        current_scale = 1;
                    exponents[(plane * rows + row) * groups + work % groups] = current_exponent;
                }
            }
            __syncthreads();
            valid = valid && !bad_scale;
            const float scale = current_scale;
            const float value = __fdiv_rn(original, scale);
            const float low = fminf(limit, fmaxf(-limit, floorf(value)));
            const float high = fminf(limit, fmaxf(-limit, ceilf(value)));
            const float nearest = fminf(limit, fmaxf(-limit, nearbyintf(value)));
            const float low_error = __fmul_rn(__fsub_rn(low, value), scale);
            const float high_error = __fmul_rn(__fsub_rn(high, value), scale);
            const float nearest_error = __fmul_rn(__fsub_rn(nearest, value), scale);
            float low_state[Rank];
            float high_state[Rank];
            float low_cost = 0;
            float high_cost = 0;
            const float d = diagonal[channel * planes + plane];
            valid = valid && isfinite(d) && d >= 0;
#pragma unroll
            for (unsigned rank = 0; rank < Rank; ++rank) {
                const float direction = factor[(channel * planes + plane) * Rank + rank];
                valid = valid && isfinite(direction);
                low_state[rank] = __fadd_rn(state[rank], __fmul_rn(low_error, direction));
                high_state[rank] = __fadd_rn(state[rank], __fmul_rn(high_error, direction));
                low_cost = __fadd_rn(low_cost, __fmul_rn(low_state[rank], low_state[rank]));
                high_cost = __fadd_rn(high_cost, __fmul_rn(high_state[rank], high_state[rank]));
                nearest_state[rank] =
                    __fadd_rn(nearest_state[rank], __fmul_rn(nearest_error, direction));
            }
            low_cost = __fadd_rn(low_cost, __fmul_rn(d, __fmul_rn(low_error, low_error)));
            high_cost = __fadd_rn(high_cost, __fmul_rn(d, __fmul_rn(high_error, high_error)));
            const bool choose_low = low_cost <= high_cost;
            const float error = choose_low ? low_error : high_error;
            candidate[plane] = static_cast<std::int8_t>(choose_low ? low : high);
#pragma unroll
            for (unsigned rank = 0; rank < Rank; ++rank)
                state[rank] = choose_low ? low_state[rank] : high_state[rank];
            candidate_diagonal =
                __fadd_rn(candidate_diagonal, __fmul_rn(d, __fmul_rn(error, error)));
            nearest_diagonal =
                __fadd_rn(nearest_diagonal, __fmul_rn(d, __fmul_rn(nearest_error, nearest_error)));
            valid = valid && isfinite(low_cost) && isfinite(high_cost) &&
                isfinite(candidate_diagonal) && isfinite(nearest_diagonal);
            __syncthreads();
        }
        float candidate_norm = 0;
        float nearest_norm = 0;
#pragma unroll
        for (unsigned rank = 0; rank < Rank; ++rank) {
            candidate_norm = __fadd_rn(candidate_norm, __fmul_rn(state[rank], state[rank]));
            nearest_norm =
                __fadd_rn(nearest_norm, __fmul_rn(nearest_state[rank], nearest_state[rank]));
        }
        const float candidate_cost = __fadd_rn(candidate_diagonal, candidate_norm);
        const float nearest_cost = __fadd_rn(nearest_diagonal, nearest_norm);
        valid = valid && isfinite(candidate_cost) && isfinite(nearest_cost);
        const bool choose_candidate = candidate_cost < nearest_cost;
        selected[row * channels + channel] = valid && choose_candidate;
        if (!valid)
            atomicAdd(invalid, 1ull);
        for (std::int64_t plane = 0; plane < planes; ++plane) {
            const auto index = (plane * rows + row) * channels + channel;
            const auto exponent = exponents[(plane * rows + row) * groups + work % groups];
            const float scale = ldexpf(1.0f, exponent);
            const float nearest =
                fminf(limit, fmaxf(-limit, nearbyintf(__fdiv_rn(source[index], scale))));
            const float code = choose_candidate ? float(candidate[plane]) : nearest;
            output[index] = valid ? __fmul_rn(code, scale) : 0;
        }
        __syncthreads();
    }
}

} // namespace

void plane_rounding(const float* source, const float* factor, const float* diagonal,
    float* reconstructed, std::int32_t* exponents, std::uint8_t* selected, std::uint64_t* invalid,
    std::int64_t planes, std::int64_t rows, std::int64_t channels, unsigned rank, unsigned bits,
    unsigned group, cudaStream_t stream)
{
    if (planes < 1 || planes > 36 || rows < 0 || channels < 1 || (rank != 1 && rank != 4) ||
        (bits != 4 && bits != 8) || (group != 32 && group != 128 && group != 256) ||
        channels % group)
        throw std::invalid_argument(
            "plane rounding requires valid shapes, rank1/4, INT4/8 and aligned groups"
        );
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    if (channels > maximum / planes / rank || rows > maximum / planes / channels)
        throw std::invalid_argument("plane rounding shape product overflows int64");
    if (rows == 0)
        return;
    if (!source || !factor || !diagonal || !reconstructed || !exponents || !selected || !invalid)
        throw std::invalid_argument("plane rounding requires valid nonaliasing device pointers");
    const auto blocks =
        static_cast<unsigned>(std::min<std::int64_t>(rows * (channels / group), 65535));
    const float limit = float((1u << (bits - 1)) - 1);
    if (rank == 1)
        rounding_kernel<1><<<blocks, group, 0, stream>>>(source, factor, diagonal, reconstructed,
            exponents, selected, reinterpret_cast<unsigned long long*>(invalid), planes, rows,
            channels, limit, group);
    else
        rounding_kernel<4><<<blocks, group, 0, stream>>>(source, factor, diagonal, reconstructed,
            exponents, selected, reinterpret_cast<unsigned long long*>(invalid), planes, rows,
            channels, limit, group);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("plane rounding kernel: ") +
            cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
