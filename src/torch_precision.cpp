#include "tensor_inference_engine/selection.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <torch/library.h>

namespace {

void validate(const at::Tensor& x, const at::Tensor& gram, double tolerance)
{
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 32 == 0 && x.is_contiguous() &&
            x.scalar_type() == at::kBFloat16,
        "adaptive precision requires contiguous BF16 [M,K], K divisible by32");
    TORCH_CHECK(gram.dim() == 2 && gram.size(0) == x.size(1) / 4 && gram.size(1) == 10 &&
            gram.device() == x.device() && gram.is_contiguous() && gram.scalar_type() == at::kFloat,
        "adaptive precision requires packed FP32 Gram [K/4,10] on the input device");
    TORCH_CHECK(std::isfinite(tolerance) && tolerance > 0, "positive finite tolerance required");
    TORCH_CHECK(!at::GradMode::is_enabled() || !(x.requires_grad() || gram.requires_grad()),
        "adaptive precision is inference-only");
}

std::tuple<at::Tensor, at::Tensor> precision_geometry_cuda(const at::Tensor& x,
    const at::Tensor& gram, double tolerance, bool fp4, std::int64_t geometry)
{
    validate(x, gram, tolerance);
    TORCH_CHECK(geometry >= 0 && geometry <= 3, "precision geometry must be0/1/2/3");
    const c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty_like(x);
    auto modes =
        at::empty({(x.size(0) + 127) / 128, (x.size(1) + 127) / 128}, x.options().dtype(at::kByte));
    tensor_inference_engine::adaptive_precision(
        reinterpret_cast<const std::uint16_t*>(x.const_data_ptr<at::BFloat16>()),
        gram.const_data_ptr<float>(),
        reinterpret_cast<std::uint16_t*>(output.mutable_data_ptr<at::BFloat16>()),
        modes.mutable_data_ptr<std::uint8_t>(), x.size(0), x.size(1), tolerance, fp4,
        static_cast<tensor_inference_engine::PrecisionGeometry>(geometry),
        c10::cuda::getCurrentCUDAStream(x.get_device()).stream()
    );
    return {output, modes};
}

std::tuple<at::Tensor, at::Tensor> precision_cuda(const at::Tensor& x, const at::Tensor& gram,
    double tolerance, bool fp4, bool diagonal)
{
    return precision_geometry_cuda(x, gram, tolerance, fp4, diagonal ? 1 : 0);
}

std::tuple<at::Tensor, at::Tensor> precision_meta(const at::Tensor& x, const at::Tensor& gram,
    double tolerance, bool, bool)
{
    validate(x, gram, tolerance);
    return {at::empty_like(x),
        at::empty({(x.size(0) + 127) / 128, (x.size(1) + 127) / 128},
            x.options().dtype(at::kByte))};
}

std::tuple<at::Tensor, at::Tensor> precision_geometry_meta(const at::Tensor& x,
    const at::Tensor& gram, double tolerance, bool fp4, std::int64_t geometry)
{
    TORCH_CHECK(geometry >= 0 && geometry <= 3, "precision geometry must be0/1/2/3");
    return precision_meta(x, gram, tolerance, fp4, false);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("adaptive_precision(Tensor x, Tensor gram, float tolerance=0.01, bool fp4=True, "
               "bool diagonal=False) -> (Tensor, Tensor)");
    module.def("adaptive_precision_geometry(Tensor x, Tensor gram, float tolerance=0.01, bool "
               "fp4=True, int geometry=0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("adaptive_precision", precision_cuda);
    module.impl("adaptive_precision_geometry", precision_geometry_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("adaptive_precision", precision_meta);
    module.impl("adaptive_precision_geometry", precision_geometry_meta);
}
