#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace tensor_inference_engine {

enum class Layout { row_major, column_major };
enum class OutputType { bf16, fp32 };

// A borrowed device matrix. BF16 values use the standard 16-bit encoding.
// leading_dimension is measured in elements, not bytes.
struct Bf16Matrix {
  const std::uint16_t* data;
  std::int64_t rows;
  std::int64_t columns;
  std::int64_t leading_dimension;
  Layout layout = Layout::row_major;
};

struct GemmOutput {
  void* data;
  std::int64_t leading_dimension;
  OutputType type = OutputType::bf16;
};

// Compute out = A B with FP32 accumulation and one final output conversion.
// Output is row-major and must not alias either input. All pointers belong to
// the caller's current CUDA device and must remain alive until stream completes.
// No allocation, device synchronization, or stream switch is performed.
// Invalid descriptors throw std::invalid_argument; launch failures throw
// std::runtime_error. Asynchronous CUDA errors surface at caller synchronization.
void gemm(Bf16Matrix a, Bf16Matrix b, GemmOutput out, cudaStream_t stream = nullptr);

}  // namespace tensor_inference_engine
