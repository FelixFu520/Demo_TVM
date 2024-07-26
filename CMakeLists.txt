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

# Add TVM include directories
include_directories(SYSTEM "${TVM_ROOT}/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/dlpack/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/dmlc-core/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/rang/include")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/compiler-rt")
include_directories(SYSTEM "${TVM_ROOT}/3rdparty/picojson")
# Add TVM library directories
link_directories("${TVM_ROOT}/build/Release")
# Add Logging library
# add_definitions("-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>")

# Find Opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Copy Files
file(COPY ${TVM_ROOT}/build/libtvm_runtime.so DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${TVM_ROOT}/build/libtvm.so DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_SOURCE_DIR}/mod.so DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
# file(COPY ${CMAKE_SOURCE_DIR}/mod.json DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
# file(COPY ${CMAKE_SOURCE_DIR}/mod.params DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
# file(COPY ${CMAKE_SOURCE_DIR}/test_resnet18.so DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_SOURCE_DIR}/cat.jpg DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(demo_tvm_cpu tvm_demo_cpu.cpp)
# add_executable(demo_tvm_cuda demo_tvm_cuda.cpp)
target_link_libraries(demo_tvm_cpu PRIVATE libtvm_runtime.so ${OpenCV_LIBS})
# target_link_libraries(demo_tvm_cuda PRIVATE tvm_runtime tvm)
