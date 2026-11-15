#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cstdio>

__global__ void convert(float* result) {
  result[threadIdx.x] = float(cutlass::bfloat16_t(float(threadIdx.x) * 0.5f));
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::fprintf(stderr, "CUDA device unavailable\n");
    return 77;
  }
  float* device = nullptr;
  if (cudaMalloc(&device, 32 * sizeof(float)) != cudaSuccess) return 1;
  convert<<<1, 32>>>(device);
  float host[32];
  auto status = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    return 1;
  }
  for (int i = 0; i < 32; ++i) if (host[i] != i * 0.5f) return 1;
  std::puts("CUDA runtime and CUTLASS BF16 conversion passed");
}
