#include "tensor_inference_engine/observation.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace {

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

at::Tensor norm_meta(const at::Tensor& values, bool columns)
{
    TORCH_CHECK(values.dim() == 2 && values.is_contiguous() &&
            values.scalar_type() == at::kBFloat16,
        "norms require a contiguous BF16 matrix");
    TORCH_CHECK(!at::GradMode::is_enabled() || !values.requires_grad(), "norms are inference-only");
    return at::empty({columns ? values.size(1) : values.size(0), 4},
        values.options().dtype(at::kFloat));
}

at::Tensor norm_cuda(const at::Tensor& values, bool columns)
{
    const c10::cuda::CUDAGuard guard(values.device());
    auto result = norm_meta(values, columns);
    tensor_inference_engine::bf16_norms(
        reinterpret_cast<const std::uint16_t*>(values.const_data_ptr<at::BFloat16>()),
        values.size(0), values.size(1), columns, result.mutable_data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream(values.get_device()).stream()
    );
    return result;
}

Result observe_meta(const at::Tensor& product, const at::Tensor& a, const at::Tensor& b,
    double coefficient, double underflow, const std::optional<at::Tensor>& bias)
{
    TORCH_CHECK(product.dim() == 2 && product.is_contiguous() &&
            product.scalar_type() == at::kFloat,
        "observation requires a contiguous FP32 matrix");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == product.size(0) &&
            b.size(0) == product.size(1) && a.size(1) == 4 && b.size(1) == 4 && a.is_contiguous() &&
            b.is_contiguous() && a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat &&
            a.device() == product.device() && b.device() == product.device(),
        "norms require matching contiguous FP32 [rows,4] storage");
    TORCH_CHECK(coefficient >= 0 && underflow >= 0 && std::isfinite(coefficient) &&
            std::isfinite(underflow),
        "finite nonnegative bound coefficients required");
    if (bias)
        TORCH_CHECK(bias->dim() == 1 && bias->numel() == product.size(1) && bias->is_contiguous() &&
                bias->scalar_type() == at::kFloat && bias->device() == product.device(),
            "bias requires matching contiguous FP32 storage");
    TORCH_CHECK(!at::GradMode::is_enabled() ||
            !(product.requires_grad() || a.requires_grad() || b.requires_grad() ||
                (bias && bias->requires_grad())),
        "observation is inference-only");
    return {at::empty_like(product, product.options().dtype(at::kBFloat16)),
        at::empty_like(product, product.options().dtype(at::kBool)), at::empty_like(product)};
}

float upper(double value)
{
    if (value == 0)
        return 0;
    return std::nextafter(static_cast<float>(value), std::numeric_limits<float>::infinity());
}

Result observe_cuda(const at::Tensor& product, const at::Tensor& a, const at::Tensor& b,
    double coefficient, double underflow, const std::optional<at::Tensor>& bias)
{
    const c10::cuda::CUDAGuard guard(product.device());
    auto result = observe_meta(product, a, b, coefficient, underflow, bias);
    tensor_inference_engine::bf16_observe(product.const_data_ptr<float>(),
        a.const_data_ptr<float>(), b.const_data_ptr<float>(),
        bias ? bias->const_data_ptr<float>() : nullptr,
        reinterpret_cast<std::uint16_t*>(std::get<0>(result).mutable_data_ptr<at::BFloat16>()),
        std::get<1>(result).mutable_data_ptr<bool>(), std::get<2>(result).mutable_data_ptr<float>(),
        product.size(0), product.size(1), upper(coefficient), upper(underflow),
        c10::cuda::getCurrentCUDAStream(product.get_device()).stream());
    return result;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("bf16_norms(Tensor values, bool columns=False) -> Tensor");
    module.def(
        "bf16_observe(Tensor product, Tensor source_norms, Tensor weight_norms, float coefficient, "
        "float underflow, Tensor? bias=None) -> (Tensor, Tensor, Tensor)"
    );
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("bf16_norms", norm_cuda);
    module.impl("bf16_observe", observe_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("bf16_norms", norm_meta);
    module.impl("bf16_observe", observe_meta);
}
