target_sources(
    tensor_inference_engine
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src/plane_rounding.cu"
        "${PROJECT_SOURCE_DIR}/src/plane_shift_rounding.cu"
)
if(BUILD_TESTING)
    add_executable(
        plane_rounding_native
        "${PROJECT_SOURCE_DIR}/tests/plane_rounding_native.cu"
    )
    target_link_libraries(plane_rounding_native PRIVATE tensor_inference_engine)
    add_test(NAME plane_rounding_native COMMAND plane_rounding_native)
    set_tests_properties(plane_rounding_native PROPERTIES SKIP_RETURN_CODE 77)
endif()
