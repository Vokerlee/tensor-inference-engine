#include "tensor_inference_engine/ordered_gemm.hpp"

#include <climits>
#include <cuda_bf16.h>
#include <stdexcept>

namespace tensor_inference_engine {
namespace {

__device__ float read(Bf16Matrix matrix, std::int64_t row, std::int64_t column)
{
    const auto offset = matrix.layout == Layout::row_major
        ? row * matrix.leading_dimension + column
        : column * matrix.leading_dimension + row;
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(matrix.data)[offset]);
}

template<unsigned Accumulators>
__global__ void kernel(Bf16Matrix a, Bf16Matrix b, float* output, std::int64_t stride,
    unsigned primitive_k, const std::int64_t* indices, std::int64_t count,
    unsigned long long* invalid)
{
    const auto position = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position >= count)
        return;
    const auto index = indices ? indices[position] : position;
    if (index < 0 || index >= a.rows * b.columns ||
        (indices && position && index <= indices[position - 1])) {
        atomicAdd(invalid, 1ULL);
        return;
    }
    const auto row = index / b.columns;
    const auto column = index % b.columns;
    float total = 0;
    for (std::int64_t begin = 0; begin < a.columns; begin += primitive_k) {
        float sums[Accumulators] = {};
        const auto end = min(a.columns, begin + primitive_k);
        for (auto k = begin; k < end; k += Accumulators) {
#pragma unroll
            for (unsigned lane = 0; lane < Accumulators; ++lane)
                if (k + lane < end)
                    sums[lane] =
                        __fmaf_rn(read(a, row, k + lane), read(b, k + lane, column), sums[lane]);
        }
        float partial;
        if constexpr (Accumulators == 1)
            partial = sums[0];
        else
            partial = __fadd_rn(__fadd_rn(sums[0], sums[1]), __fadd_rn(sums[2], sums[3]));
        total = __fadd_rn(total, partial);
    }
    output[row * stride + column] = total;
}

void validate(Bf16Matrix x)
{
    if (x.rows < 0 || x.columns < 0 || x.rows > INT_MAX || x.columns > INT_MAX ||
        x.leading_dimension < 0 || x.leading_dimension > INT_MAX ||
        (x.layout != Layout::row_major && x.layout != Layout::column_major))
        throw std::invalid_argument("invalid ordered GEMM dimensions/layout");
    const auto required = x.layout == Layout::row_major ? x.columns : x.rows;
    if (x.rows && x.columns && (!x.data || x.leading_dimension < required))
        throw std::invalid_argument("invalid ordered GEMM data/stride");
}

} // namespace

void ordered_gemm(Bf16Matrix a, Bf16Matrix b, GemmOutput out, unsigned primitive_k,
    unsigned accumulators, const std::int64_t* indices, std::int64_t count,
    std::uint64_t* invalid_count, cudaStream_t stream)
{
    validate(a);
    validate(b);
    if (a.columns != b.rows || !primitive_k || (accumulators != 1 && accumulators != 4) ||
        out.type != OutputType::fp32 || out.leading_dimension < b.columns ||
        out.leading_dimension > INT_MAX || count < 0)
        throw std::invalid_argument("invalid ordered GEMM contract/output");
    if (!indices)
        count = a.rows * b.columns;
    if (!count)
        return;
    if (!out.data || !invalid_count || (count + 255) / 256 > INT_MAX)
        throw std::invalid_argument("invalid ordered GEMM output/count storage");
    const auto grid = static_cast<unsigned>((count + 255) / 256);
    auto launch = [&]<unsigned Chains>() {
        kernel<Chains><<<grid, 256, 0, stream>>>(a, b, static_cast<float*>(out.data),
            out.leading_dimension, primitive_k, indices, count,
            reinterpret_cast<unsigned long long*>(invalid_count));
    };
    if (accumulators == 1)
        launch.operator()<1>();
    else
        launch.operator()<4>();
    if (auto status = cudaGetLastError(); status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

} // namespace tensor_inference_engine
