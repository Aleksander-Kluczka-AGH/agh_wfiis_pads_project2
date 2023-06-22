cmake_minimum_required(VERSION 3.11)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type: Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if ("$ENV{UPCXX_GASNET_CONDUIT}" STREQUAL "")
    set(ENV{UPCXX_GASNET_CONDUIT} udp)
endif()

project(daps_project2_2023 CXX C)

option(ENABLE_LOGGING "Print debug logs" False)
option(ENABLE_ASAN "Enable address sanitizer" False)

set(UPCXX_PATH "/opt/nfs/berkeley_upcxx-2023.3.0")
set(UPCXX_BINARY_PATH "${UPCXX_PATH}/bin")
set(UPCXX_INCLUDE_PATH "${UPCXX_PATH}/include")
set(UPCXX_COMPILER "${UPCXX_BINARY_PATH}/upcxx")
set(UPCXX_RUNNER "${UPCXX_BINARY_PATH}/upcxx-run")
set(PARALLEL_TARGETS_FILE "${PROJECT_SOURCE_DIR}/build/nodes")
set(SHARED_HEAP_SIZE "256M")
set(PARALLEL_PROCESSES_COUNT 17)
# set(UPCXX_CODEMODE opt)

set(CMAKE_VERBOSE_MAKEFILE False)
set(CMAKE_CXX_COMPILER "${UPCXX_COMPILER}")

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/bin" OR NOT IS_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
    message(STATUS "Creating binary directory '${PROJECT_SOURCE_DIR}/bin'")
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
endif()

add_custom_command(
    OUTPUT ${PARALLEL_TARGETS_FILE}
    COMMAND /opt/nfs/config/station204_name_list.sh
    ARGS 1 16 > ${PARALLEL_TARGETS_FILE}
    COMMENT "Generating parallel targets file: '${PARALLEL_TARGETS_FILE}'"
)
add_library(parallel_nodes OBJECT ${PARALLEL_TARGETS_FILE})

###

add_executable(fft_parallel)
set_target_properties(fft_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_sources(fft_parallel PRIVATE "${PROJECT_SOURCE_DIR}/fft_parallel.cpp")
target_compile_definitions(fft_parallel PRIVATE
    $<$<BOOL:${ENABLE_LOGGING}>:ENABLE_LOGGING>
)
target_compile_features(fft_parallel PRIVATE c_std_23 cxx_std_20)
target_compile_options(fft_parallel PRIVATE $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>)
target_include_directories(fft_parallel PRIVATE "${UPCXX_INCLUDE_PATH}")
target_link_options(fft_parallel PRIVATE $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>)

add_custom_target(run
    ${UPCXX_RUNNER} -shared-heap ${SHARED_HEAP_SIZE}
                    -n ${PARALLEL_PROCESSES_COUNT}
                    `upcxx-nodes ${PARALLEL_TARGETS_FILE}`
                    $<TARGET_FILE:fft_parallel>
    DEPENDS $<TARGET_FILE:fft_parallel>
            parallel_nodes
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running parallel FFT algorithm..."
)

###

add_executable(fft_sequential)
set_target_properties(fft_sequential PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_sources(fft_sequential PRIVATE "${PROJECT_SOURCE_DIR}/fft_sequential.cpp")
target_compile_features(fft_sequential PRIVATE c_std_23 cxx_std_20)
target_link_libraries(fft_sequential PRIVATE m)

add_custom_target(sequential $<TARGET_FILE:fft_sequential>
    DEPENDS $<TARGET_FILE:fft_sequential>
)
