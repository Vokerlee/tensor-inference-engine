cmake_minimum_required(VERSION 3.24)

if(
    NOT DEFINED CMAKE_CUDA_COMPILER
    AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.toolchains/cuda-12.8.1/bin/nvcc"
)
    set(CMAKE_CUDA_COMPILER
        "${CMAKE_CURRENT_SOURCE_DIR}/.toolchains/cuda-12.8.1/bin/nvcc"
        CACHE FILEPATH
        "CUDA compiler"
    )
endif()
set(CMAKE_CUDA_ARCHITECTURES
    89
    CACHE STRING
    "CUDA code generation architectures"
)
project(tensor_inference_engine VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.8 REQUIRED)
get_target_property(tie_cudart_location CUDA::cudart IMPORTED_LOCATION)
get_filename_component(tie_cuda_library_dir "${tie_cudart_location}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH "${tie_cuda_library_dir}")

include(FetchContent)
FetchContent_Declare(
    cutlass
    URL
        https://github.com/NVIDIA/cutlass/archive/098de2a652cf8f00fd70b2df54051c7eccbb855a.tar.gz
    URL_HASH
        SHA256=a8a1b952c7f6179bc8076f20d45859f18bd81f21f3d22b6766d7f3f0cceb5d54
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(cutlass)
add_library(tie_cutlass INTERFACE)
target_include_directories(
    tie_cutlass
    SYSTEM
    INTERFACE "${cutlass_SOURCE_DIR}/include"
)

include(CTest)
option(TIE_BUILD_TORCH "Build the optional Torch operator adapter" OFF)
if(BUILD_TESTING)
    add_executable(cuda_smoke tests/cuda_smoke.cu)
    target_link_libraries(cuda_smoke PRIVATE CUDA::cudart tie_cutlass)
    add_test(NAME cuda_smoke COMMAND cuda_smoke)
    set_tests_properties(cuda_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Operator modules remain independently reviewable additions to the core build.
include(cmake/Gemm.cmake OPTIONAL)
include(cmake/Torch.cmake OPTIONAL)
