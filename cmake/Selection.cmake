target_sources(
    tensor_inference_engine
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src/selection.cu"
        "${PROJECT_SOURCE_DIR}/src/preparation.cu"
        "${PROJECT_SOURCE_DIR}/src/precision.cu"
)
