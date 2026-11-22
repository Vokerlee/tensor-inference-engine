if(NOT TIE_BUILD_TORCH)
    return()
endif()
if(NOT TARGET tensor_inference_engine)
    message(FATAL_ERROR "The Torch adapter requires the GEMM core")
endif()
find_package(Python3 3.12 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND
        "${Python3_EXECUTABLE}" -c
        "import torch; print(torch.__path__[0]); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE tie_torch_info
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE tie_torch_result
)
if(NOT tie_torch_result EQUAL 0)
    message(
        FATAL_ERROR
        "Install requirements.txt into the selected Python environment"
    )
endif()
string(REPLACE "\n" ";" tie_torch_info "${tie_torch_info}")
list(GET tie_torch_info 0 tie_torch_root)
list(GET tie_torch_info 1 tie_torch_abi)

add_library(
    tensor_inference_engine_torch
    SHARED
    "${PROJECT_SOURCE_DIR}/src/torch.cpp"
    "${PROJECT_SOURCE_DIR}/src/torch_selection.cpp"
    "${PROJECT_SOURCE_DIR}/src/torch_preparation.cpp"
)
target_include_directories(
    tensor_inference_engine_torch
    SYSTEM
    PRIVATE
        "${tie_torch_root}/include"
        "${tie_torch_root}/include/torch/csrc/api/include"
)
target_compile_definitions(
    tensor_inference_engine_torch
    PRIVATE _GLIBCXX_USE_CXX11_ABI=${tie_torch_abi}
)
target_link_libraries(
    tensor_inference_engine_torch
    PRIVATE tensor_inference_engine
)
foreach(
    library
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
)
    set(location "${tie_torch_root}/lib/lib${library}.so")
    if(NOT EXISTS "${location}")
        message(FATAL_ERROR "Selected Torch installation lacks ${location}")
    endif()
    target_link_libraries(tensor_inference_engine_torch PRIVATE "${location}")
endforeach()
set_target_properties(
    tensor_inference_engine_torch
    PROPERTIES BUILD_RPATH "${tie_torch_root}/lib;${tie_cuda_library_dir}"
)
file(
    GENERATE OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/tensor_inference_engine_build.json"
    CONTENT
        "{\n  \"cuda_compiler_version\": \"${CMAKE_CUDA_COMPILER_VERSION}\",\n  \"cuda_architectures\": \"${CMAKE_CUDA_ARCHITECTURES}\",\n  \"cxx_compiler_id\": \"${CMAKE_CXX_COMPILER_ID}\",\n  \"cxx_compiler_version\": \"${CMAKE_CXX_COMPILER_VERSION}\",\n  \"build_type\": \"${CMAKE_BUILD_TYPE}\",\n  \"torch_cxx11_abi\": ${tie_torch_abi}\n}\n"
)
