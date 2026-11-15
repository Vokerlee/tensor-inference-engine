add_library(tensor_inference_engine STATIC "${PROJECT_SOURCE_DIR}/src/gemm.cu")
add_library(tensor_inference_engine::core ALIAS tensor_inference_engine)
set_target_properties(
    tensor_inference_engine
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    tensor_inference_engine
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(
    tensor_inference_engine
    PUBLIC CUDA::cudart
    PRIVATE tie_cutlass
)
target_compile_options(
    tensor_inference_engine
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(BUILD_TESTING)
    add_executable(gemm_native "${PROJECT_SOURCE_DIR}/tests/gemm_native.cu")
    target_link_libraries(
        gemm_native
        PRIVATE tensor_inference_engine tie_cutlass
    )
    add_test(NAME gemm_native COMMAND gemm_native)
    set_tests_properties(gemm_native PROPERTIES SKIP_RETURN_CODE 77)
endif()
