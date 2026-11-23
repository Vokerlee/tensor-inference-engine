#include "tensor_inference_engine/integer_segments.hpp"
#include "tensor_inference_engine/integer_segments_cuda.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

template<bool UseL1>
__global__ void plan_kernel(const std::int32_t* row_max, const std::int32_t* row_exponent,
    const std::int32_t* window_l1, const std::int32_t* window_low, const std::int32_t* window_high,
    const std::int32_t* row_l1, const std::int32_t* window_max, std::uint8_t* ends,
    std::uint8_t* oversized, bool* valid, std::int64_t count, std::int64_t windows,
    std::int64_t groups, std::int64_t slices, int groups_per_slice, unsigned bits)
{
    for (std::int64_t index = blockIdx.x * std::int64_t(blockDim.x) + threadIdx.x; index < count;
        index += gridDim.x * std::int64_t(blockDim.x)) {
        const auto slice = index % slices;
        const auto window = (index / slices) % windows;
        const auto row = index / (slices * windows);
        IntegerGroupBound bounds[8];
        unsigned length = 0;
        bool metadata_valid = true;
        for (int local = 0; local < groups_per_slice; ++local) {
            const auto group = slice * groups_per_slice + local;
            if (group >= groups)
                break;
            const auto a = row * groups + group;
            const auto w = window * groups + group;
            const auto maximum = row_max[a];
            const auto weight_l1 = window_l1[w];
            metadata_valid &= maximum >= 0 && weight_l1 >= 0;
            std::int32_t activation_l1 = 0;
            std::int32_t weight_maximum = 0;
            if constexpr (UseL1) {
                activation_l1 = row_l1[a];
                weight_maximum = window_max[w];
                metadata_valid &=
                    activation_l1 >= maximum && weight_maximum >= 0 && weight_l1 >= weight_maximum;
            }
            if (maximum <= 0 || weight_l1 <= 0) {
                bounds[length++] = {0, 0, 0};
                continue;
            }
            auto magnitude = std::uint64_t(maximum) * std::uint64_t(weight_l1);
            if constexpr (UseL1) {
                if (activation_l1 < 0 || weight_maximum < 0) {
                    bounds[length++] = {0, 0, 0};
                    continue;
                }
                const auto alternative =
                    std::uint64_t(activation_l1) * std::uint64_t(weight_maximum);
                magnitude = magnitude < alternative ? magnitude : alternative;
            }
            bounds[length++] = {magnitude, std::int64_t(row_exponent[a]) + window_low[w],
                std::int64_t(row_exponent[a]) + window_high[w]};
        }
        auto plan = plan_integer_segments(bounds, length, bits);
        plan.valid &= metadata_valid;
        if (!plan.valid) {
            plan.ends = static_cast<std::uint8_t>((1u << length) - 1);
            plan.oversized = plan.ends;
        }
        ends[index] = plan.ends;
        oversized[index] = plan.oversized;
        valid[index] = plan.valid;
    }
}

} // namespace

void plan_dot_integer_segments(const std::int32_t* row_max, const std::int32_t* row_exponent,
    const std::int32_t* window_l1, const std::int32_t* window_low, const std::int32_t* window_high,
    const std::int32_t* row_l1, const std::int32_t* window_max, std::uint8_t* ends,
    std::uint8_t* oversized, bool* valid, std::int64_t rows, std::int64_t windows,
    std::int64_t groups, int groups_per_slice, unsigned bits, cudaStream_t stream)
{
    if (rows < 0 || windows < 0 || groups < 0 || groups_per_slice < 1 || groups_per_slice > 8 ||
        bits < 2 || bits > 48 || ((row_l1 == nullptr) != (window_max == nullptr)))
        throw std::invalid_argument(
            "invalid integer segment dimensions, width or optional statistics"
        );
    if (!rows || !windows || !groups)
        return;
    const auto slices = (groups + groups_per_slice - 1) / groups_per_slice;
    const auto count = rows * windows * slices;
    constexpr int threads = 128;
    const auto blocks =
        static_cast<int>(std::min<std::int64_t>((count + threads - 1) / threads, 65535));
    if (row_l1)
        plan_kernel<true><<<blocks, threads, 0, stream>>>(row_max, row_exponent, window_l1,
            window_low, window_high, row_l1, window_max, ends, oversized, valid, count, windows,
            groups, slices, groups_per_slice, bits);
    else
        plan_kernel<false><<<blocks, threads, 0, stream>>>(row_max, row_exponent, window_l1,
            window_low, window_high, nullptr, nullptr, ends, oversized, valid, count, windows,
            groups, slices, groups_per_slice, bits);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("integer segment kernel: ") +
            cudaGetErrorString(error));
}

} // namespace tensor_inference_engine
