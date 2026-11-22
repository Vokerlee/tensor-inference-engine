#include "tensor_inference_engine/selection.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

namespace {

void validate(const at::Tensor& x, const at::Tensor& q, const at::Tensor& gram)
{
    TORCH_CHECK(x.dim() == 2 && q.sizes() == x.sizes() && x.size(1) % 4 == 0,
        "selection requires equal [M, K] shapes with K divisible by four");
    TORCH_CHECK(gram.dim() == 2 && gram.size(0) == x.size(1) / 4 && gram.size(1) == 10,
        "selection requires packed Gram [K/4, 10]");
    for (const auto* tensor : {&x, &q, &gram}) {
        TORCH_CHECK(tensor->scalar_type() == at::kFloat && tensor->is_contiguous(),
            "selection requires contiguous FP32 tensors");
        TORCH_CHECK(tensor->device() == x.device(), "selection tensors must share a device");
        TORCH_CHECK(!at::GradMode::is_enabled() || !tensor->requires_grad(),
            "selection is inference-only");
    }
}

std::tuple<at::Tensor, at::Tensor> select_cuda(const at::Tensor& x, const at::Tensor& q,
    const at::Tensor& gram, bool diagonal)
{
    validate(x, q, gram);
    const c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty_like(x);
    auto pairs = at::empty({x.size(0), x.size(1) / 4}, x.options().dtype(at::kByte));
    tensor_inference_engine::select_2of4(x.const_data_ptr<float>(), q.const_data_ptr<float>(),
        gram.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
        pairs.mutable_data_ptr<std::uint8_t>(), x.size(0), x.size(1), diagonal,
        c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return {output, pairs};
}

std::tuple<at::Tensor, at::Tensor> select_meta(const at::Tensor& x, const at::Tensor& q,
    const at::Tensor& gram, bool)
{
    validate(x, q, gram);
    return {at::empty_like(x), at::empty({x.size(0), x.size(1) / 4}, x.options().dtype(at::kByte))};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("select_2of4(Tensor x, Tensor reconstructed, Tensor gram, bool diagonal=False) -> "
               "(Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("select_2of4", select_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("select_2of4", select_meta);
}
