cmake_minimum_required (VERSION 3.6)

project (GPGPU_2017)
set (CMAKE_BUILD_TYPE Release)

# set output directories
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

# cuda
find_package (CUDA REQUIRED 7.5)
if (NOT CUDA_FOUND)
    message (FATAL_ERROR "Fatal error: CUDA SDK (version >= 7.5) required.")
endif (NOT CUDA_FOUND)
# configure cuda
#   compile for baseline of CC 2.0, and archtecture 2.0/3.0
list (APPEND CUDA_NVCC_FLAGS "-arch=compute_30; -code=sm_30; -std=c++11")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/utils")

# configure compiler
add_compile_options (-Wall)
add_compile_options (-std=c++14)
include_directories ("${PROJECT_SOURCE_DIR}/utils")

# build demo program
add_subdirectory ("lab0" "lab1" "lab2")
