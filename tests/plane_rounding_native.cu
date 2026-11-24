#include "tensor_inference_engine/plane_rounding.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    constexpr int planes = 4;
    constexpr int channels = 256;
    float *source, *factor, *diagonal, *output;
    std::int32_t* exponents;
    std::uint8_t* selected;
    std::uint64_t* invalid;
    check(cudaMalloc(&source, planes * channels * sizeof(float)));
    check(cudaMalloc(&factor, planes * channels * sizeof(float)));
    check(cudaMalloc(&diagonal, planes * channels * sizeof(float)));
    check(cudaMalloc(&output, planes * channels * sizeof(float)));
    check(cudaMalloc(&exponents, planes * sizeof(std::int32_t)));
    check(cudaMalloc(&selected, channels));
    check(cudaMalloc(&invalid, sizeof(std::uint64_t)));
    std::vector<float> values(planes * channels, 0.4f);
    std::vector<float> directions(planes * channels, 1);
    check(cudaMemcpy(source, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(factor, directions.data(), directions.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    check(cudaMemset(diagonal, 0, values.size() * sizeof(float)));
    check(cudaMemset(invalid, 0, sizeof(std::uint64_t)));
    tensor_inference_engine::plane_rounding(source, factor, diagonal, output, exponents, selected,
        invalid, planes, 1, channels, 1, 4, channels);
    check(cudaDeviceSynchronize());
    std::vector<float> result(values.size());
    std::vector<std::uint8_t> choices(channels);
    std::uint64_t bad;
    check(cudaMemcpy(result.data(), output, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(choices.data(), selected, choices.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&bad, invalid, sizeof(bad), cudaMemcpyDeviceToHost));
    for (int plane = 0; plane < planes; ++plane)
        for (int channel = 0; channel < channels; ++channel)
            if (result[plane * channels + channel] != (plane % 2 ? 0.4375f : 0.375f) ||
                choices[channel] != 1 || bad != 0)
                throw std::runtime_error("independent scalar cancellation witness failed");
    std::int16_t* metadata;
    check(cudaMalloc(&metadata, values.size() * sizeof(std::int16_t)));
    std::vector<std::int16_t> packed(values.size(), 257);
    check(cudaMemcpy(metadata, packed.data(), packed.size() * sizeof(std::int16_t),
        cudaMemcpyHostToDevice));
    check(cudaMemset(invalid, 0, sizeof(std::uint64_t)));
    tensor_inference_engine::plane_shift_rounding(source, metadata, output, exponents, selected,
        invalid, planes, 1, channels, 4, channels);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(), output, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(choices.data(), selected, choices.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&bad, invalid, sizeof(bad), cudaMemcpyDeviceToHost));
    for (int plane = 0; plane < planes; ++plane)
        for (int channel = 0; channel < channels; ++channel)
            if (result[plane * channels + channel] != (plane % 2 ? 0.4375f : 0.375f) ||
                choices[channel] != 1 || bad != 0)
                throw std::runtime_error("causal shift cancellation witness failed");
    check(cudaFree(metadata));
    for (auto pointer : {source, factor, diagonal, output})
        check(cudaFree(pointer));
    check(cudaFree(exponents));
    check(cudaFree(selected));
    check(cudaFree(invalid));
    std::cout << "Bounded plane rounding cancellation witness passed\n";
}
