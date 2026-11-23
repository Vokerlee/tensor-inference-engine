target_sources(
    tensor_inference_engine
    PRIVATE "${PROJECT_SOURCE_DIR}/src/integer_segments.cu"
)

if(BUILD_TESTING)
    add_executable(integer_segments_native tests/integer_segments.cpp)
    target_include_directories(
        integer_segments_native
        PRIVATE "${PROJECT_SOURCE_DIR}/include"
    )
    add_test(NAME integer_segments_native COMMAND integer_segments_native)

    add_executable(integer_segments_cuda tests/integer_segments_cuda.cu)
    target_link_libraries(integer_segments_cuda PRIVATE tensor_inference_engine)
    add_test(NAME integer_segments_cuda COMMAND integer_segments_cuda)
    set_tests_properties(integer_segments_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()
