if (NOT MSVC)
    set(CMAKE_CXX_STANDARD 14)
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# build TensorRT ops
if ("trt" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    if (NOT DEFINED TENSORRT_DIR)
        set(TENSORRT_DIR $ENV{TENSORRT_DIR})
    endif ()
    message(STATUS "Build TensorRT custom ops.")
    add_subdirectory(tensorrt)
endif ()
