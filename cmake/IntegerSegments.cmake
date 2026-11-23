if(BUILD_TESTING)
    add_executable(integer_segments_native tests/integer_segments.cpp)
    target_include_directories(
        integer_segments_native
        PRIVATE "${PROJECT_SOURCE_DIR}/include"
    )
    add_test(NAME integer_segments_native COMMAND integer_segments_native)
endif()
