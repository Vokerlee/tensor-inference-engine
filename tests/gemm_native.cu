#include "tensor_inference_engine/gemm.hpp"
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace tie = tensor_inference_engine;
using Bf16 = cutlass::bfloat16_t;

void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void* data = nullptr;

    explicit Buffer(std::size_t bytes)
    {
        check(cudaMalloc(&data, std::max(bytes, std::size_t(1))));
    }

    ~Buffer() { cudaFree(data); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

std::size_t index(int row, int column, int ld, tie::Layout layout)
{
    return layout == tie::Layout::row_major ? row * std::size_t(ld) + column
                                            : column * std::size_t(ld) + row;
}

template<typename Output>
void run(int m, int n, int k, tie::Layout la, tie::Layout lb, int padding, cudaStream_t stream)
{
    int lda = (la == tie::Layout::row_major ? k : m) + padding;
    int ldb = (lb == tie::Layout::row_major ? n : k) + padding;
    int ldc = n + padding;
    std::vector<Bf16> a((la == tie::Layout::row_major ? m : k) * std::size_t(lda));
    std::vector<Bf16> b((lb == tie::Layout::row_major ? k : n) * std::size_t(ldb));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            a[index(i, j, lda, la)] = Bf16(float((i * 13 + j * 7) % 17 - 8) / 8);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j)
            b[index(i, j, ldb, lb)] = Bf16(float((i * 11 + j * 3) % 19 - 9) / 8);
    std::vector<Output> out(m * std::size_t(ldc), Output(42.f));
    Buffer da(a.size() * sizeof(Bf16)), db(b.size() * sizeof(Bf16)),
        dc(out.size() * sizeof(Output));
    if (!a.empty())
        check(cudaMemcpyAsync(da.data, a.data(), a.size() * sizeof(Bf16), cudaMemcpyHostToDevice,
            stream));
    if (!b.empty())
        check(cudaMemcpyAsync(db.data, b.data(), b.size() * sizeof(Bf16), cudaMemcpyHostToDevice,
            stream));
    if (!out.empty())
        check(cudaMemcpyAsync(dc.data, out.data(), out.size() * sizeof(Output),
            cudaMemcpyHostToDevice, stream));
    auto type = sizeof(Output) == 2 ? tie::OutputType::bf16 : tie::OutputType::fp32;
    tie::gemm({static_cast<std::uint16_t*>(da.data), m, k, lda, la},
        {static_cast<std::uint16_t*>(db.data), k, n, ldb, lb}, {dc.data, ldc, type}, stream);
    if (!out.empty())
        check(cudaMemcpyAsync(out.data(), dc.data, out.size() * sizeof(Output),
            cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j) {
            double reference = 42;
            if (j < n) {
                reference = 0;
                for (int t = 0; t < k; ++t)
                    reference +=
                        double(float(a[index(i, t, lda, la)])) * float(b[index(t, j, ldb, lb)]);
            }
            if (float(out[i * ldc + j]) != float(Output(float(reference))))
                throw std::runtime_error(
                    "GEMM differs from exact binary-fraction reference or overwrites row padding"
                );
        }
}

int main()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    try {
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int shapes[][3] = {
            {0,   7,   4  },
            {8,   0,   4  },
            {5,   7,   0  },
            {1,   1,   1  },
            {3,   5,   7  },
            {65,  47,  33 },
            {17,  19,  8  },
            {128, 128, 128},
            {33,  65,  16 },
            {256, 128, 64 },
            {257, 513, 8  }
        };
        int cases = 0;
        for (auto& shape : shapes)
            for (auto la : {tie::Layout::row_major, tie::Layout::column_major})
                for (auto lb : {tie::Layout::row_major, tie::Layout::column_major})
                    for (int pad : {0, 3}) {
                        run<Bf16>(shape[0], shape[1], shape[2], la, lb, pad, stream);
                        run<float>(shape[0], shape[1], shape[2], la, lb, pad, stream);
                        cases += 2;
                    }
        check(cudaStreamDestroy(stream));
        auto expect_invalid = [](tie::Bf16Matrix a, tie::Bf16Matrix b, tie::GemmOutput c) {
            try {
                tie::gemm(a, b, c);
            } catch (const std::invalid_argument&) {
                return;
            }
            throw std::runtime_error("invalid GEMM descriptor accepted");
        };
        expect_invalid({nullptr, -1, 1, 1}, {nullptr, 1, 1, 1}, {nullptr, 1});
        expect_invalid({nullptr, 0, 1, 1}, {nullptr, 2, 0, 0}, {nullptr, 0});
        expect_invalid({nullptr, 0, 1, 1}, {nullptr, 1, 0, 0},
            {nullptr, 0, static_cast<tie::OutputType>(99)});
        std::printf("%d GEMM shape/layout/output cases and invalid-descriptor checks passed\n",
            cases);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
