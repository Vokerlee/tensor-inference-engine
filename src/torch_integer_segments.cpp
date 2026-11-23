#include "tensor_inference_engine/integer_segments_cuda.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <optional>
#include <tuple>
#include <vector>

namespace {

using OptionalTensor = std::optional<at::Tensor>;
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

void same_metadata(const at::Tensor& values, const at::Tensor& reference)
{
    TORCH_CHECK(values.sizes() == reference.sizes() && values.device() == reference.device() &&
            values.is_contiguous() && values.scalar_type() == at::kInt,
        "segment metadata must be contiguous int32 with matching shape and device");
}

Result allocate(const at::Tensor& maximum, const at::Tensor& exponent, const at::Tensor& weight_l1,
    const at::Tensor& low, const at::Tensor& high, std::int64_t groups_per_slice, std::int64_t bits,
    const OptionalTensor& l1, const OptionalTensor& weight_maximum)
{
    TORCH_CHECK(maximum.dim() == 2 && weight_l1.dim() == 2 && maximum.size(1) == weight_l1.size(1),
        "segment statistics require matching [rows,groups] and [windows,groups]");
    same_metadata(maximum, maximum);
    same_metadata(exponent, maximum);
    same_metadata(weight_l1, weight_l1);
    same_metadata(low, weight_l1);
    same_metadata(high, weight_l1);
    TORCH_CHECK(maximum.device() == weight_l1.device(), "segment statistics require one device");
    TORCH_CHECK(groups_per_slice >= 1 && groups_per_slice <= 8 && bits >= 2 && bits <= 48,
        "segment planning requires one to eight groups per slice and width2..48");
    TORCH_CHECK(l1.has_value() == weight_maximum.has_value(),
        "both optional norm tensors are required");
    if (l1) {
        same_metadata(*l1, maximum);
        same_metadata(*weight_maximum, weight_l1);
    }
    const auto slices = (maximum.size(1) + groups_per_slice - 1) / groups_per_slice;
    const auto shape = std::vector<std::int64_t>{maximum.size(0), weight_l1.size(0), slices};
    return {at::empty(shape, maximum.options().dtype(at::kByte)),
        at::empty(shape, maximum.options().dtype(at::kByte)),
        at::empty(shape, maximum.options().dtype(at::kBool))};
}

Result plan_cuda(const at::Tensor& maximum, const at::Tensor& exponent, const at::Tensor& weight_l1,
    const at::Tensor& low, const at::Tensor& high, std::int64_t groups_per_slice, std::int64_t bits,
    const OptionalTensor& l1, const OptionalTensor& weight_maximum)
{
    const c10::cuda::CUDAGuard guard(maximum.device());
    auto result = allocate(maximum, exponent, weight_l1, low, high, groups_per_slice, bits, l1,
        weight_maximum);
    tensor_inference_engine::plan_dot_integer_segments(maximum.const_data_ptr<std::int32_t>(),
        exponent.const_data_ptr<std::int32_t>(), weight_l1.const_data_ptr<std::int32_t>(),
        low.const_data_ptr<std::int32_t>(), high.const_data_ptr<std::int32_t>(),
        l1 ? l1->const_data_ptr<std::int32_t>() : nullptr,
        weight_maximum ? weight_maximum->const_data_ptr<std::int32_t>() : nullptr,
        std::get<0>(result).mutable_data_ptr<std::uint8_t>(),
        std::get<1>(result).mutable_data_ptr<std::uint8_t>(),
        std::get<2>(result).mutable_data_ptr<bool>(), maximum.size(0), weight_l1.size(0),
        maximum.size(1), groups_per_slice, bits,
        c10::cuda::getCurrentCUDAStream(maximum.get_device()).stream());
    return result;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(tensor_inference_engine, module)
{
    module.def("integer_segment_plan(Tensor maximum, Tensor exponent, Tensor weight_l1, "
               "Tensor low, Tensor high, int groups_per_slice=4, int bits=24, "
               "Tensor? l1=None, Tensor? weight_maximum=None) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module)
{
    module.impl("integer_segment_plan", plan_cuda);
}

TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module)
{
    module.impl("integer_segment_plan", allocate);
}
