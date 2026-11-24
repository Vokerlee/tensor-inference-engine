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

// Full outputs share operand tiles. Sparse outputs use the compact index
// kernel above; this keeps the full-reference comparison independently useful.
template<unsigned Accumulators>
__global__ void tiled_kernel(Bf16Matrix a, Bf16Matrix b, float* output, std::int64_t stride,
    unsigned primitive_k, std::int64_t column_tiles)
{
    constexpr unsigned Rows = 4, Columns = 64, Reduction = 32;
    __shared__ float source[Rows][Reduction];
    __shared__ float weight[Reduction][Columns];
    const auto row_base = (std::int64_t(blockIdx.x) / column_tiles) * Rows;
    const auto column_base = (std::int64_t(blockIdx.x) % column_tiles) * Columns;
    const auto local_row = threadIdx.x / Columns;
    const auto local_column = threadIdx.x % Columns;
    float total = 0;
    for (std::int64_t begin = 0; begin < a.columns; begin += primitive_k) {
        const auto end = min(a.columns, begin + primitive_k);
        float sums[Accumulators] = {};
        for (auto tile = begin; tile < end; tile += Reduction) {
            for (unsigned index = threadIdx.x; index < Rows * Reduction; index += blockDim.x) {
                const auto row = row_base + index / Reduction;
                const auto k = tile + index % Reduction;
                source[index / Reduction][index % Reduction] =
                    row < a.rows && k < end ? read(a, row, k) : 0;
            }
            for (unsigned index = threadIdx.x; index < Reduction * Columns; index += blockDim.x) {
                const auto k = tile + index / Columns;
                const auto column = column_base + index % Columns;
                weight[index / Columns][index % Columns] =
                    k < end && column < b.columns ? read(b, k, column) : 0;
            }
            __syncthreads();
            for (unsigned offset = 0; offset < Reduction; offset += Accumulators) {
#pragma unroll
                for (unsigned lane = 0; lane < Accumulators; ++lane)
                    if (tile + offset + lane < end)
                        sums[lane] = __fmaf_rn(source[local_row][offset + lane],
                            weight[offset + lane][local_column], sums[lane]);
            }
            __syncthreads();
        }
        float partial;
        if constexpr (Accumulators == 1)
            partial = sums[0];
        else
            partial = __fadd_rn(__fadd_rn(sums[0], sums[1]), __fadd_rn(sums[2], sums[3]));
        total = __fadd_rn(total, partial);
    }
    if (row_base + local_row < a.rows && column_base + local_column < b.columns)
        output[(row_base + local_row) * stride + column_base + local_column] = total;
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
    std::uint64_t* invalid_count, cudaStream_t stream, bool tiled)
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
    const auto columns = (b.columns + 63) / 64;
    const auto tiles = ((a.rows + 3) / 4) * columns;
    if (!indices && tiled && tiles > INT_MAX)
        throw std::invalid_argument("ordered GEMM tile grid exceeds int32");
    const auto grid = static_cast<unsigned>(indices || !tiled ? (count + 255) / 256 : tiles);
    auto launch = [&]<unsigned Chains>() {
        if (indices || !tiled)
            kernel<Chains><<<grid, 256, 0, stream>>>(a, b, static_cast<float*>(out.data),
                out.leading_dimension, primitive_k, indices, count,
                reinterpret_cast<unsigned long long*>(invalid_count));
        else
            tiled_kernel<Chains><<<grid, 256, 0, stream>>>(a, b, static_cast<float*>(out.data),
                out.leading_dimension, primitive_k, columns);
    };
    if (accumulators == 1)
        launch.operator()<1>();
    else
        launch.operator()<4>();
    if (auto status = cudaGetLastError(); status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

} // namespace tensor_inference_engine
