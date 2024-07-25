# CMake minimal version
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Project name
project(demo_tvm C CXX)

# C++ 17
set(CMAKE_CXX_STANDARD 17)
# C++ 17 required
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(TVM_ROOT "${CMAKE_SOURCE_DIR}/third_party/tvm")
set(DMLC_CORE "${TVM_ROOT}/3rdparty/dmlc-core")

include_directories(SYSTEM "${TVM_ROOT}/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/dlpack/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/dmlc-core/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/rang/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/compiler-rt")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/picojson")

link_directories("${TVM_ROOT}/build/Release")

add_definitions("-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>")

add_executable(demo_tvm main.cpp)
target_link_libraries(demo_tvm PRIVATE tvm_runtime tvm)
