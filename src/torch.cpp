#include "tensor_inference_engine/gemm.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

namespace tie = tensor_inference_engine;
namespace {
void validate(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm requires two-dimensional matrices");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
              "gemm requires BF16 operands");
  TORCH_CHECK(a.layout() == at::kStrided && b.layout() == at::kStrided,
              "gemm requires strided tensors");
  TORCH_CHECK(a.device() == b.device(), "gemm operands must share a device");
  TORCH_CHECK(a.sym_size(1) == b.sym_size(0), "gemm inner dimensions differ");
}

tie::Bf16Matrix view(const at::Tensor& x) {
  auto data = reinterpret_cast<const std::uint16_t*>(x.const_data_ptr<at::BFloat16>());
  if (x.numel() == 0) return {data, x.size(0), x.size(1), 0};
  if (x.stride(1) == 1 && x.stride(0) >= x.size(1))
    return {data, x.size(0), x.size(1), x.stride(0), tie::Layout::row_major};
  if (x.stride(0) == 1 && x.stride(1) >= x.size(0))
    return {data, x.size(0), x.size(1), x.stride(1), tie::Layout::column_major};
  TORCH_CHECK(false, "gemm supports row/column-major inputs with non-overlapping leading dimensions");
}

at::Tensor gemm_cuda(const at::Tensor& a, const at::Tensor& b, bool output_fp32) {
  validate(a, b);
  TORCH_CHECK(a.is_cuda(), "gemm requires CUDA operands");
  TORCH_CHECK(!at::GradMode::is_enabled() || !(a.requires_grad() || b.requires_grad()),
              "gemm is inference-only; use torch.inference_mode() or disable gradients");
  const c10::cuda::CUDAGuard guard(a.device());
  auto av = view(a);
  auto bv = view(b);
  auto result = at::empty({a.size(0), b.size(1)},
                         a.options().dtype(output_fp32 ? at::kFloat : at::kBFloat16));
  tie::gemm(av, bv, {result.mutable_data_ptr(), result.stride(0),
            output_fp32 ? tie::OutputType::fp32 : tie::OutputType::bf16},
            c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
  return result;
}

at::Tensor gemm_meta(const at::Tensor& a, const at::Tensor& b, bool output_fp32) {
  validate(a, b);
  return at::empty_symint({a.sym_size(0), b.sym_size(1)},
      a.options().device(at::kMeta).dtype(output_fp32 ? at::kFloat : at::kBFloat16));
}
}  // namespace

TORCH_LIBRARY(tensor_inference_engine, module) {
  module.def("gemm(Tensor a, Tensor b, bool output_fp32=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(tensor_inference_engine, CUDA, module) {
  module.impl("gemm", gemm_cuda);
}
TORCH_LIBRARY_IMPL(tensor_inference_engine, Meta, module) {
  module.impl("gemm", gemm_meta);
}
