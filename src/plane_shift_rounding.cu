#include "tensor_inference_engine/plane_rounding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

__device__ float maximum_warp(float value)
{
    for (int offset = 16; offset; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__global__ void shift_kernel(const float* source, const std::int16_t* metadata, float* output,
    std::int32_t* exponents, std::uint8_t* changed, unsigned long long* invalid,
    std::int64_t planes, std::int64_t rows, std::int64_t channels, float limit, unsigned group)
{
    __shared__ float maxima[8];
    __shared__ int exponent;
    __shared__ int bad_scale;
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    const auto groups = channels / group;
    for (std::int64_t work = blockIdx.x; work < rows * groups; work += gridDim.x) {
        const auto row = work / groups;
        const auto channel = (work % groups) * group + threadIdx.x;
        float state = 0;
        bool valid = true;
        bool different = false;
        for (std::int64_t plane = 0; plane < planes; ++plane) {
            const auto index = (plane * rows + row) * channels + channel;
            const float loaded = source[index];
            valid = valid && isfinite(loaded);
            const float original = isfinite(loaded) ? loaded : 0;
            const float reduced = maximum_warp(fabsf(original));
            if (lane == 0)
                maxima[warp] = reduced;
            __syncthreads();
            if (warp == 0) {
                const float maximum = maximum_warp(lane < group / 32 ? maxima[lane] : 0);
                if (lane == 0) {
                    exponent = 0;
                    bad_scale = 0;
                    if (maximum > 0) {
                        const float logarithm = log2f(__fdiv_rn(maximum, limit));
                        bad_scale = !isfinite(logarithm);
                        if (!bad_scale) {
                            exponent = static_cast<int>(ceilf(logarithm));
                            const float scale = ldexpf(1.0f, exponent);
                            bad_scale = !isfinite(scale) || scale <= 0;
                        }
                    }
                    exponents[(plane * rows + row) * groups + work % groups] = exponent;
                }
            }
            __syncthreads();
            valid = valid && !bad_scale;
            const int packed = metadata[channel * planes + plane];
            const int raw_direction = packed & 255;
            const int direction = raw_direction >= 128 ? raw_direction - 256 : raw_direction;
            const int diagonal = (packed >> 8) & 255;
            valid = valid && packed >= 0 && direction != -128 && diagonal <= 127;
            const int direction_exponent = 1 - abs(direction);
            const float scaled = ldexpf(original, -exponent);
            const float lower = fminf(limit, fmaxf(-limit, floorf(scaled)));
            const float upper = fminf(limit, fmaxf(-limit, ceilf(scaled)));
            const float nearest = fminf(limit, fmaxf(-limit, nearbyintf(scaled)));
            const float low_value = ldexpf(lower, exponent);
            const float high_value = ldexpf(upper, exponent);
            const float low_error = __fsub_rn(low_value, original);
            const float high_error = __fsub_rn(high_value, original);
            float projected_state = direction == 0 ? 0 : ldexpf(state, direction_exponent);
            if (direction < 0)
                projected_state = -projected_state;
            const float diagonal_term =
                diagonal == 0 ? 0 : ldexpf(__fadd_rn(low_error, high_error), 1 - diagonal);
            const float criterion = __fadd_rn(ldexpf(projected_state, 1), diagonal_term);
            const bool choose_low = criterion > 0 || (criterion == 0 && nearest == lower);
            const float code = choose_low ? lower : upper;
            const float error = choose_low ? low_error : high_error;
            float update = direction == 0 ? 0 : ldexpf(error, direction_exponent);
            if (direction < 0)
                update = -update;
            state = __fadd_rn(state, update);
            valid = valid && isfinite(criterion) && isfinite(state) && isfinite(low_value) &&
                isfinite(high_value);
            different = different || code != nearest;
            output[index] = valid ? (choose_low ? low_value : high_value) : 0;
            __syncthreads();
        }
        changed[row * channels + channel] = valid && different;
        if (!valid)
            atomicAdd(invalid, 1ull);
        __syncthreads();
    }
}

} // namespace

void plane_shift_rounding(const float* source, const std::int16_t* metadata, float* reconstructed,
    std::int32_t* exponents, std::uint8_t* changed, std::uint64_t* invalid, std::int64_t planes,
    std::int64_t rows, std::int64_t channels, unsigned bits, unsigned group, cudaStream_t stream)
{
    if (planes < 1 || planes > 36 || rows < 0 || channels < 1 || (bits != 4 && bits != 8) ||
        (group != 32 && group != 128 && group != 256) || channels % group)
        throw std::invalid_argument(
            "shift rounding requires valid shapes, INT4/8 and aligned groups"
        );
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    if (channels > maximum / planes || rows > maximum / planes / channels)
        throw std::invalid_argument("shift rounding shape product overflows int64");
    if (!rows)
        return;
    if (!source || !metadata || !reconstructed || !exponents || !changed || !invalid)
        throw std::invalid_argument("shift rounding requires valid nonaliasing device pointers");
    const auto blocks =
        static_cast<unsigned>(std::min<std::int64_t>(rows * (channels / group), 65535));
    shift_kernel<<<blocks, group, 0, stream>>>(source, metadata, reconstructed, exponents, changed,
        reinterpret_cast<unsigned long long*>(invalid), planes, rows, channels,
        float((1u << (bits - 1)) - 1), group);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("shift rounding kernel: ") +
            cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
