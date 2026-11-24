#include "tensor_inference_engine/plane_rounding.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <tuple>

namespace {

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

Result allocate(const at::Tensor& values, const at::Tensor& factor, const at::Tensor& diagonal,
    std::int64_t bits, std::int64_t group, bool check_finite)
{
    TORCH_CHECK(values.dim() == 3 && values.is_contiguous() && values.scalar_type() == at::kFloat &&
            values.size(0) >= 1 && values.size(0) <= 36 && values.size(2) > 0,
        "plane rounding requires contiguous FP32 [planes,rows,channels], planes1..36");
    TORCH_CHECK((group == 32 || group == 128 || group == 256) && values.size(2) % group == 0,
        "plane rounding requires group32/128/256 aligned channels");
    TORCH_CHECK(bits == 4 || bits == 8, "plane rounding supports INT4/8");
    TORCH_CHECK(factor.dim() == 3 && factor.size(0) == values.size(2) &&
            factor.size(1) == values.size(0) && (factor.size(2) == 1 || factor.size(2) == 4) &&
            factor.is_contiguous() && factor.scalar_type() == at::kFloat &&
            factor.device() == values.device(),
        "plane factor requires contiguous FP32 [channels,planes,rank1/4] on the same device");
    TORCH_CHECK(diagonal.dim() == 2 && diagonal.size(0) == values.size(2) &&
            diagonal.size(1) == values.size(0) && diagonal.is_contiguous() &&
            diagonal.scalar_type() == at::kFloat && diagonal.device() == values.device(),
        "plane diagonal requires contiguous FP32 [channels,planes] on the same device");
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            !(values.requires_grad() || factor.requires_grad() || diagonal.requires_grad()),
        "plane rounding is inference-only");
    return {at::empty_like(values),
        at::empty({values.size(0), values.size(1), values.size(2) / group},
            values.options().dtype(at::kInt)),
        at::empty({values.size(1), values.size(2)}, values.options().dtype(at::kByte)),
        at::zeros({1}, values.options().dtype(at::kLong))};
}

Result round_cuda(const at::Tensor& values, const at::Tensor& factor, const at::Tensor& diagonal,
    std::int64_t bits, std::int64_t group, bool check_finite)
{
    const c10::cuda::CUDAGuard guard(values.device());
    auto result = allocate(values, factor, diagonal, bits, group, check_finite);
    tensor_inference_engine::plane_rounding(values.const_data_ptr<float>(),
        factor.const_data_ptr<float>(), diagonal.const_data_ptr<float>(),
        std::get<0>(result).mutable_data_ptr<float>(),
        std::get<1>(result).mutable_data_ptr<std::int32_t>(),
        std::get<2>(result).mutable_data_ptr<std::uint8_t>(),
        reinterpret_cast<std::uint64_t*>(std::get<3>(result).mutable_data_ptr<std::int64_t>()),
        values.size(0), values.size(1), values.size(2), static_cast<unsigned>(factor.size(2)),
        static_cast<unsigned>(bits), static_cast<unsigned>(group),
        c10::cuda::getCurrentCUDAStream(values.get_device()).stream());
    TORCH_CHECK(
        !check_finite || std::get<3>(result).item<std::int64_t>() == 0,
        "plane rounding requires finite source/metadata, nonnegative diagonal and finite arithmetic"
    );
    return result;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("plane_rounding(Tensor values, Tensor factor, Tensor diagonal, int bits=4, "
               "int group=32, bool check_finite=True) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("plane_rounding", round_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("plane_rounding", allocate);
}
