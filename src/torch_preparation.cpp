#include "tensor_inference_engine/selection.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <optional>

namespace {

using OptionalTensor = std::optional<at::Tensor>;

void validate(const at::Tensor& x, const OptionalTensor& gram, std::int64_t bits,
    std::int64_t policy)
{
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 32 == 0 && x.is_contiguous() &&
            x.scalar_type() == at::kBFloat16,
        "preparation requires contiguous BF16 [M,K] with K divisible by32");
    TORCH_CHECK((bits == 4 || bits == 8) && policy >= 0 && policy <= 2,
        "preparation supports bits4/8 and policy0/1/2");
    TORCH_CHECK(!at::GradMode::is_enabled() || !x.requires_grad(), "preparation is inference-only");
    TORCH_CHECK(policy == 0 || gram.has_value(), "sparse preparation requires packed Gram");
    if (gram) {
        TORCH_CHECK(gram->dim() == 2 && gram->size(0) == x.size(1) / 4 && gram->size(1) == 10 &&
                gram->device() == x.device() && gram->is_contiguous() &&
                gram->scalar_type() == at::kFloat,
            "preparation requires contiguous FP32 packed Gram [K/4,10] on the input device");
        TORCH_CHECK(!at::GradMode::is_enabled() || !gram->requires_grad(),
            "preparation is inference-only");
    }
}

std::tuple<at::Tensor, at::Tensor> prepare_cuda(const at::Tensor& x, const OptionalTensor& gram,
    std::int64_t bits, std::int64_t policy)
{
    validate(x, gram, bits, policy);
    const c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty_like(x);
    auto pairs = at::empty({x.size(0), x.size(1) / 4}, x.options().dtype(at::kByte));
    tensor_inference_engine::prepare_activations(
        reinterpret_cast<const std::uint16_t*>(x.const_data_ptr<at::BFloat16>()),
        gram ? gram->const_data_ptr<float>() : nullptr,
        reinterpret_cast<std::uint16_t*>(output.mutable_data_ptr<at::BFloat16>()),
        pairs.mutable_data_ptr<std::uint8_t>(), x.size(0), x.size(1), bits, policy,
        c10::cuda::getCurrentCUDAStream(x.get_device()).stream()
    );
    return {output, pairs};
}

std::tuple<at::Tensor, at::Tensor> prepare_meta(const at::Tensor& x, const OptionalTensor& gram,
    std::int64_t bits, std::int64_t policy)
{
    validate(x, gram, bits, policy);
    return {at::empty_like(x), at::empty({x.size(0), x.size(1) / 4}, x.options().dtype(at::kByte))};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("prepare_activations(Tensor x, Tensor? gram=None, int bits=4, int policy=0) -> "
               "(Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("prepare_activations", prepare_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("prepare_activations", prepare_meta);
}
