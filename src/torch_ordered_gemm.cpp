#include "tensor_inference_engine/ordered_gemm.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <climits>
#include <tuple>

namespace {

using Result = std::tuple<at::Tensor, at::Tensor>;

Result allocate(const at::Tensor& a, const at::Tensor& b, std::int64_t primitive_k,
    std::int64_t accumulators, const std::optional<at::Tensor>& indices,
    const std::optional<at::Tensor>& initial, bool check_indices, bool tiled)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
        "ordered GEMM matrix shapes differ");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
            a.is_contiguous() && b.is_contiguous() && a.device() == b.device(),
        "ordered GEMM requires contiguous BF16 matrices on one device");
    TORCH_CHECK(primitive_k > 0 && primitive_k <= INT_MAX &&
            (accumulators == 1 || accumulators == 4),
        "ordered GEMM requires positive primitive_k and one or four accumulators");
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            !(a.requires_grad() || b.requires_grad() || (initial && initial->requires_grad())),
        "ordered GEMM is inference-only");
    if (indices) {
        TORCH_CHECK(indices->dim() == 1 && indices->is_contiguous() &&
                indices->scalar_type() == at::kLong && indices->device() == a.device(),
            "indices require contiguous INT64 on the operand device");
        TORCH_CHECK(initial.has_value(), "indexed ordered GEMM requires an initial output");
    }
    if (initial)
        TORCH_CHECK(initial->dim() == 2 && initial->size(0) == a.size(0) &&
                initial->size(1) == b.size(1) && initial->is_contiguous() &&
                initial->scalar_type() == at::kFloat && initial->device() == a.device(),
            "initial output must be contiguous FP32 with matching shape/device");
    auto options = a.options().dtype(at::kFloat);
    return {initial ? initial->clone() : at::empty({a.size(0), b.size(1)}, options),
        at::zeros({1}, a.options().dtype(at::kLong))};
}

Result execute(const at::Tensor& a, const at::Tensor& b, std::int64_t primitive_k,
    std::int64_t accumulators, const std::optional<at::Tensor>& indices,
    const std::optional<at::Tensor>& initial, bool check_indices, bool tiled)
{
    const c10::cuda::CUDAGuard guard(a.device());
    auto result = allocate(a, b, primitive_k, accumulators, indices, initial, check_indices, tiled);
    if (indices && !indices->numel())
        return result;
    namespace tie = tensor_inference_engine;
    tie::ordered_gemm({reinterpret_cast<const std::uint16_t*>(a.const_data_ptr<at::BFloat16>()),
                          a.size(0), a.size(1), a.stride(0)},
        {reinterpret_cast<const std::uint16_t*>(b.const_data_ptr<at::BFloat16>()), b.size(0),
            b.size(1), b.stride(0)},
        {std::get<0>(result).mutable_data_ptr(), b.size(1), tie::OutputType::fp32}, primitive_k,
        accumulators, indices ? indices->const_data_ptr<std::int64_t>() : nullptr,
        indices ? indices->numel() : a.size(0) * b.size(1),
        reinterpret_cast<std::uint64_t*>(std::get<1>(result).mutable_data_ptr<std::int64_t>()),
        c10::cuda::getCurrentCUDAStream(a.get_device()).stream(), tiled);
    TORCH_CHECK(!indices || !check_indices || std::get<1>(result).item<std::int64_t>() == 0,
        "ordered GEMM indices must be in bounds and strictly increasing");
    return result;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("ordered_gemm(Tensor a, Tensor b, int primitive_k=128, int accumulators=1, "
               "Tensor? indices=None, Tensor? initial=None, bool check_indices=True, bool "
               "tiled=False) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("ordered_gemm", execute);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("ordered_gemm", allocate);
}
