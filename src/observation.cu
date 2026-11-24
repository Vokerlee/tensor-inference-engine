#include "tensor_inference_engine/observation.hpp"

#include <cfloat>
#include <climits>
#include <cuda_bf16.h>
#include <stdexcept>

namespace tensor_inference_engine {
namespace {

__global__ void norms_kernel(const std::uint16_t* values, std::int64_t rows, std::int64_t columns,
    bool along_columns, float* output)
{
    const auto index = std::int64_t(blockIdx.x);
    const auto reduction = along_columns ? rows : columns;
    const auto lane = threadIdx.x;
    float sum = 0, maximum = 0, squares = 0;
    unsigned bad = 0;
    for (auto k = std::int64_t(lane); k < reduction; k += 32) {
        const auto word = values[along_columns ? k * columns + index : index * columns + k];
        const auto absolute = word & 0x7fff;
        bad |= (absolute && absolute < 0x80) || absolute >= 0x7f80;
        const float value =
            absolute >= 0x7f80 ? 0 : fabsf(__bfloat162float(__ushort_as_bfloat16(word)));
        sum = __fadd_ru(sum, value);
        squares = __fadd_ru(squares, __fmul_ru(value, value));
        maximum = fmaxf(maximum, value);
    }
    for (unsigned offset = 16; offset; offset /= 2) {
        sum = __fadd_ru(sum, __shfl_down_sync(0xffffffff, sum, offset));
        squares = __fadd_ru(squares, __shfl_down_sync(0xffffffff, squares, offset));
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        bad |= __shfl_down_sync(0xffffffff, bad, offset);
    }
    if (!lane) {
        output[index * 4] = sum;
        output[index * 4 + 1] = maximum;
        output[index * 4 + 2] = __fsqrt_ru(squares);
        output[index * 4 + 3] = bad;
    }
}

__global__ void observation_kernel(const float* product, const float* a, const float* b,
    const float* bias, std::uint16_t* observed, bool* accepted, float* errors, std::int64_t rows,
    std::int64_t columns, float coefficient, float underflow)
{
    const auto index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * columns)
        return;
    const auto row = index / columns, column = index % columns;
    a += row * 4;
    b += column * 4;
    const float scale =
        fminf(fminf(__fmul_ru(a[1], b[0]), __fmul_ru(a[0], b[1])), __fmul_ru(a[2], b[2]));
    const float raw = scale == 0 ? 0 : product[index];
    const float shift = bias ? bias[column] : 0;
    const float value = bias ? __fadd_rn(raw, shift) : raw;
    float error = __fadd_ru(__fmul_ru(scale, coefficient), underflow);
    if (bias)
        error = __fadd_ru(__fadd_ru(error, __fmul_ru(error, 0x1p-24f)),
            __fadd_ru(__fmul_ru(__fadd_ru(fabsf(raw), fabsf(shift)), 0x1p-23f), 0x1p-148f));
    if (scale == 0)
        error = 0;
    const auto rounded = __float2bfloat16_rn(value);
    const auto word = __bfloat16_as_ushort(rounded);
    observed[index] = word;
    errors[index] = error;
    const auto previous_word = static_cast<unsigned short>(word & 0x8000 ? word + 1 : word - 1);
    const auto following_word = static_cast<unsigned short>(word & 0x8000 ? word - 1 : word + 1);
    const auto center = __bfloat162float(rounded);
    const auto previous = __bfloat162float(__ushort_as_bfloat16(previous_word));
    const auto following = __bfloat162float(__ushort_as_bfloat16(following_word));
    const float lower = __fmul_rn(__fadd_rn(previous, center), 0.5f);
    const float upper = __fmul_rn(__fadd_rn(following, center), 0.5f);
    const auto exponent = word & 0x7f80;
    const bool domain = !a[3] && !b[3] && isfinite(scale) && scale < FLT_MAX / 4 &&
        isfinite(shift) && __fadd_ru(scale, fabsf(shift)) < FLT_MAX / 4;
    const bool interior = exponent > 0 && exponent < 0x7f00 && isfinite(error) &&
        __fsub_rd(value, error) > lower && __fadd_ru(value, error) < upper;
    accepted[index] = domain && (interior || (scale == 0 && isfinite(value)));
}

void check()
{
    if (auto status = cudaGetLastError(); status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

} // namespace

void bf16_norms(const std::uint16_t* values, std::int64_t rows, std::int64_t columns,
    bool along_columns, float* output, cudaStream_t stream)
{
    const auto count = along_columns ? columns : rows;
    if (!count)
        return;
    if (rows < 0 || columns < 0 || count > INT_MAX || !output || (rows && columns && !values))
        throw std::invalid_argument("invalid BF16 norm storage/dimensions");
    norms_kernel<<<static_cast<unsigned>(count), 32, 0, stream>>>(values, rows, columns,
        along_columns, output);
    check();
}

void bf16_observe(const float* product, const float* source_norms, const float* weight_norms,
    const float* bias, std::uint16_t* observed, bool* accepted, float* error, std::int64_t rows,
    std::int64_t columns, float coefficient, float underflow, cudaStream_t stream)
{
    const auto count = rows * columns;
    if (!count)
        return;
    if (rows < 0 || columns < 0 || (count + 255) / 256 > INT_MAX || !product || !source_norms ||
        !weight_norms || !observed || !accepted || !error || coefficient < 0 || underflow < 0)
        throw std::invalid_argument("invalid BF16 observation descriptors");
    observation_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(product,
        source_norms, weight_norms, bias, observed, accepted, error, rows, columns, coefficient,
        underflow);
    check();
}

} // namespace tensor_inference_engine
