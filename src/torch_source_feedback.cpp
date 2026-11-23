#include "tensor_inference_engine/source_feedback.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <tuple>

namespace {

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

Result allocate(const at::Tensor& values, const at::Tensor& coefficient, std::int64_t bits,
    bool preserve_zeros, bool check_finite)
{
    TORCH_CHECK(values.dim() == 3 && values.is_contiguous() &&
            values.scalar_type() == at::kBFloat16 && values.size(2) > 0 && values.size(2) % 32 == 0,
        "feedback requires contiguous BF16 [streams,width,channels] with aligned channels");
    TORCH_CHECK(coefficient.dim() == 1 && coefficient.size(0) == values.size(2) &&
            coefficient.is_contiguous() && coefficient.scalar_type() == at::kFloat &&
            coefficient.device() == values.device(),
        "feedback coefficients require contiguous FP32 [channels] on the same device");
    TORCH_CHECK(bits == 4 || bits == 8, "feedback supports bits4/8");
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            !(values.requires_grad() || coefficient.requires_grad()),
        "feedback is inference-only");
    return {at::empty_like(values),
        at::empty({values.size(0), values.size(2)}, values.options().dtype(at::kFloat)),
        at::empty({values.size(0), values.size(1), values.size(2) / 32},
            values.options().dtype(at::kInt)),
        at::zeros({3}, values.options().dtype(at::kLong))};
}

Result feedback_cuda(const at::Tensor& values, const at::Tensor& coefficient, std::int64_t bits,
    bool preserve_zeros, bool check_finite)
{
    const c10::cuda::CUDAGuard guard(values.device());
    auto result = allocate(values, coefficient, bits, preserve_zeros, check_finite);
    tensor_inference_engine::source_feedback(
        reinterpret_cast<const std::uint16_t*>(values.const_data_ptr<at::BFloat16>()),
        coefficient.const_data_ptr<float>(),
        reinterpret_cast<std::uint16_t*>(std::get<0>(result).mutable_data_ptr<at::BFloat16>()),
        std::get<1>(result).mutable_data_ptr<float>(),
        std::get<2>(result).mutable_data_ptr<std::int32_t>(),
        reinterpret_cast<std::uint64_t*>(std::get<3>(result).mutable_data_ptr<std::int64_t>()),
        values.size(0), values.size(1), values.size(2), static_cast<unsigned>(bits), preserve_zeros,
        c10::cuda::getCurrentCUDAStream(values.get_device()).stream()
    );
    TORCH_CHECK(
        !check_finite || std::get<3>(result)[2].item<std::int64_t>() == 0,
        "feedback requires finite values/coefficients in [-1,1] and finite representable arithmetic"
    );
    return result;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("source_feedback(Tensor values, Tensor coefficient, int bits=4, bool "
               "preserve_zeros=False, bool check_finite=True) "
               "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("source_feedback", feedback_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("source_feedback", allocate);
}
