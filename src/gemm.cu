#include "tensor_inference_engine/gemm.hpp"

#include <climits>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <stdexcept>
#include <string>

namespace tensor_inference_engine {
namespace {

void validate(Bf16Matrix x)
{
    if (x.rows < 0 || x.columns < 0 || x.rows > INT_MAX || x.columns > INT_MAX ||
        x.leading_dimension < 0 || x.leading_dimension > INT_MAX)
        throw std::invalid_argument("matrix dimensions and strides must fit nonnegative int32");
    if (x.layout != Layout::row_major && x.layout != Layout::column_major)
        throw std::invalid_argument("unsupported matrix layout");
    if (x.rows && x.columns) {
        auto minimum = x.layout == Layout::row_major ? x.columns : x.rows;
        if (!x.data || x.leading_dimension < minimum)
            throw std::invalid_argument("null matrix data or overlapping leading dimension");
    }
}

template<typename LayoutA, typename LayoutB, typename Output, int Alignment, int Tile>
void launch(Bf16Matrix a, Bf16Matrix b, GemmOutput out, cudaStream_t stream)
{
    using Kernel = cutlass::gemm::device::Gemm<cutlass::bfloat16_t, LayoutA, cutlass::bfloat16_t,
        LayoutB, Output, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<Tile, Tile, 32>,
        cutlass::gemm::GemmShape<Tile / 2, Tile / 2, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<Output,
            Alignment == 8 ? 128 / cutlass::sizeof_bits<Output>::value : 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Alignment == 8 ? 3 : 2,
        Alignment, Alignment>;
    typename Kernel::Arguments arguments{
        {int(a.rows), int(b.columns), int(a.columns)},
        {reinterpret_cast<const cutlass::bfloat16_t*>(a.data), int(a.leading_dimension)},
        {reinterpret_cast<const cutlass::bfloat16_t*>(b.data), int(b.leading_dimension)},
        {static_cast<Output*>(out.data), int(out.leading_dimension)},
        {static_cast<Output*>(out.data), int(out.leading_dimension)},
        {1.f, 0.f}
    };
    auto status = Kernel::can_implement(arguments);
    if (status == cutlass::Status::kSuccess)
        status = Kernel{}(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS GEMM: ") + cutlassGetStatusString(status));
}

template<typename LayoutA, typename LayoutB, typename Output>
void dispatch_alignment(Bf16Matrix a, Bf16Matrix b, GemmOutput out, cudaStream_t stream)
{
    bool aligned = (reinterpret_cast<std::uintptr_t>(a.data) % 16 == 0) &&
        (reinterpret_cast<std::uintptr_t>(b.data) % 16 == 0) &&
        (reinterpret_cast<std::uintptr_t>(out.data) % 16 == 0) && a.leading_dimension % 8 == 0 &&
        b.leading_dimension % 8 == 0 && out.leading_dimension % 8 == 0 && a.columns % 8 == 0 &&
        b.columns % 8 == 0;
    // Small or narrow outputs need more threadblocks to occupy the GPU.
    if (a.rows <= 256 || b.columns <= 256) {
        if (aligned)
            launch<LayoutA, LayoutB, Output, 8, 64>(a, b, out, stream);
        else
            launch<LayoutA, LayoutB, Output, 1, 64>(a, b, out, stream);
    } else {
        if (aligned)
            launch<LayoutA, LayoutB, Output, 8, 128>(a, b, out, stream);
        else
            launch<LayoutA, LayoutB, Output, 1, 128>(a, b, out, stream);
    }
}

template<typename Output>
void dispatch_layout(Bf16Matrix a, Bf16Matrix b, GemmOutput out, cudaStream_t stream)
{
    using Row = cutlass::layout::RowMajor;
    using Column = cutlass::layout::ColumnMajor;
    if (a.layout == Layout::row_major) {
        if (b.layout == Layout::row_major)
            dispatch_alignment<Row, Row, Output>(a, b, out, stream);
        else
            dispatch_alignment<Row, Column, Output>(a, b, out, stream);
    } else {
        if (b.layout == Layout::row_major)
            dispatch_alignment<Column, Row, Output>(a, b, out, stream);
        else
            dispatch_alignment<Column, Column, Output>(a, b, out, stream);
    }
}

} // namespace

void gemm(Bf16Matrix a, Bf16Matrix b, GemmOutput out, cudaStream_t stream)
{
    validate(a);
    validate(b);
    if (a.columns != b.rows)
        throw std::invalid_argument("GEMM inner dimensions differ");
    if (out.type != OutputType::bf16 && out.type != OutputType::fp32)
        throw std::invalid_argument("unsupported output type");
    if (out.leading_dimension < b.columns || out.leading_dimension > INT_MAX)
        throw std::invalid_argument("invalid output leading dimension");
    if (!a.rows || !b.columns)
        return;
    if (!out.data)
        throw std::invalid_argument("null output data");
    if (!a.columns) {
        auto bytes = out.type == OutputType::bf16 ? sizeof(std::uint16_t) : sizeof(float);
        auto status = cudaMemset2DAsync(out.data, out.leading_dimension * bytes, 0,
            b.columns * bytes, a.rows, stream);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        return;
    }
    if (out.type == OutputType::bf16)
        dispatch_layout<cutlass::bfloat16_t>(a, b, out, stream);
    else
        dispatch_layout<float>(a, b, out, stream);
}

} // namespace tensor_inference_engine
