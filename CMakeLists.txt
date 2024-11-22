# Copyright (c) OpenMMLab. All rights reserved.
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "installation directory")
endif ()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 3.14)
project(MMDeploy VERSION 0.9.0)

set(CMAKE_CXX_STANDARD 17)

set(MMDEPLOY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(MMDEPLOY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(MMDEPLOY_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if (MSVC)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else ()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif ()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# options
option(MMDEPLOY_SHARED_LIBS "build shared libs" ON)
option(MMDEPLOY_ELENA_FUSION "use elena to fuse preprocess" OFF)

set(MMDEPLOY_TARGET_DEVICES "cpu" CACHE STRING "target devices to support")
# set(MMDEPLOY_TARGET_BACKENDS "" CACHE STRING "target inference engines to support")

set(MMDEPLOY_TARGET_BACKENDS "trt")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "choose 'Release' as default build type" FORCE)
endif ()

if (MMDEPLOY_SHARED_LIBS)
    set(MMDEPLOY_LIB_TYPE SHARED)
else ()
    set(MMDEPLOY_LIB_TYPE STATIC)
endif ()

set(MMDEPLOY_TASKS "" CACHE INTERNAL "")

if (MMDEPLOY_COVERAGE)
    add_compile_options(-coverage -fprofile-arcs -ftest-coverage)
    add_link_options(-coverage -lgcov)
endif ()

# when CUDA devices are enabled, the environment variable ASAN_OPTIONS=protect_shadow_gap=0
# must be set at runtime
if (MMDEPLOY_ASAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_link_options(-fsanitize=address)
endif ()


add_library(MMDeployStaticModules INTERFACE)
add_library(MMDeployDynamicModules INTERFACE)
add_library(MMDeployLibs INTERFACE)
include(cmake/cuda.cmake NO_POLICY_SCOPE)


# this must come after including cuda.cmake because policies in function scope is captured
# at function definition
include(cmake/MMDeploy.cmake)

add_subdirectory(csrc/mmdeploy)

