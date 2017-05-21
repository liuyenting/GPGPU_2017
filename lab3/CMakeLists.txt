file (GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

list (APPEND CUDA_NVCC_FLAGS "-O2")

# generate the executable
cuda_add_executable (lab3 ${SOURCES})
target_link_libraries (lab3 ${CUDA_LIBRARIES})

# add custom target to run the binary
add_custom_target (run
    COMMAND lab3
    DEPENDS lab3
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
