#include "tensor_inference_engine/integer_segments.hpp"
#include "tensor_inference_engine/integer_segments_cuda.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void check(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

struct CudaFree {
    void operator()(void* pointer) const { cudaFree(pointer); }
};

template<class T>
auto allocate(std::size_t count)
{
    T* pointer = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    return std::unique_ptr<T, CudaFree>(pointer);
}

auto upload(const std::vector<std::int32_t>& values)
{
    auto result = allocate<std::int32_t>(values.size());
    check(cudaMemcpy(result.get(), values.data(), values.size() * sizeof(std::int32_t),
        cudaMemcpyHostToDevice));
    return result;
}

void run()
{
    constexpr int rows = 3, windows = 2, groups = 9;
    std::vector<std::int32_t> maximum(rows * groups), exponent(rows * groups), l1(rows * groups);
    std::vector<std::int32_t> weight_max(windows * groups), weight_l1(windows * groups);
    std::vector<std::int32_t> low(windows * groups), high(windows * groups);
    for (int i = 0; i < rows * groups; ++i) {
        maximum[i] = i < groups ? 0 : (i * 17) % 127;
        l1[i] = maximum[i] * 16;
        exponent[i] = i % 7 - 5;
    }
    for (int i = 0; i < windows * groups; ++i) {
        weight_max[i] = i % 5 ? 100 : 0;
        weight_l1[i] = weight_max[i] * 13;
        low[i] = i % 11 - 7;
        high[i] = low[i] + i % 4;
    }
    auto a = upload(maximum), ae = upload(exponent), al1 = upload(l1);
    auto wm = upload(weight_max), wl1 = upload(weight_l1), lo = upload(low), hi = upload(high);
    for (int width : {8, 24, 48}) {
        for (int per_slice : {4, 8}) {
            const auto slices = (groups + per_slice - 1) / per_slice;
            const auto count = rows * windows * slices;
            auto ends = allocate<std::uint8_t>(count), oversized = allocate<std::uint8_t>(count);
            auto valid = allocate<bool>(count);
            for (bool use_l1 : {false, true}) {
                tensor_inference_engine::plan_dot_integer_segments(a.get(), ae.get(), wl1.get(),
                    lo.get(), hi.get(), use_l1 ? al1.get() : nullptr, use_l1 ? wm.get() : nullptr,
                    ends.get(), oversized.get(), valid.get(), rows, windows, groups, per_slice,
                    width, nullptr);
                std::vector<std::uint8_t> actual_ends(count), actual_oversized(count),
                    actual_valid(count);
                static_assert(sizeof(bool) == 1);
                check(cudaMemcpy(actual_ends.data(), ends.get(), count, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(actual_oversized.data(), oversized.get(), count,
                    cudaMemcpyDeviceToHost));
                check(cudaMemcpy(actual_valid.data(), valid.get(), count, cudaMemcpyDeviceToHost));
                for (int row = 0; row < rows; ++row) {
                    for (int window = 0; window < windows; ++window) {
                        for (int slice = 0; slice < slices; ++slice) {
                            tensor_inference_engine::IntegerGroupBound bounds[8];
                            const auto length = std::min(per_slice, groups - slice * per_slice);
                            for (int local = 0; local < length; ++local) {
                                const auto group = slice * per_slice + local;
                                const auto ai = row * groups + group, wi = window * groups + group;
                                auto magnitude = std::uint64_t(maximum[ai]) * weight_l1[wi];
                                if (use_l1)
                                    magnitude =
                                        std::min(magnitude, std::uint64_t(l1[ai]) * weight_max[wi]);
                                bounds[local] = {magnitude, exponent[ai] + low[wi],
                                    exponent[ai] + high[wi]};
                            }
                            const auto expected =
                                tensor_inference_engine::plan_integer_segments(bounds, length,
                                    width);
                            const auto index = (row * windows + window) * slices + slice;
                            if (actual_ends[index] != expected.ends ||
                                actual_oversized[index] != expected.oversized ||
                                actual_valid[index] != expected.valid)
                                throw std::runtime_error(
                                    "fused CUDA segment plan differs from scalar reference"
                                );
                        }
                    }
                }
            }
        }
    }
}

} // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    try {
        run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
