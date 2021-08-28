cmake_minimum_required(VERSION 3.18)
project(tsqr-tc LANGUAGES CXX CUDA)

find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)
set(LIBDIR lib)

##########################################################################
# Library
##########################################################################

# Source files
file(GLOB SRCS "${SRCDIR}/*.cu")
file(GLOB HEADERS "${INCDIR}/*.hpp")

add_library(tsqr-tc STATIC
	${SRCS}
	${HEADERS}
	)

# Include directories
target_include_directories(tsqr-tc PUBLIC ${INCDIR})
target_include_directories(tsqr-tc PUBLIC ${SRCDIR}/cutf/include)
target_include_directories(tsqr-tc PUBLIC ${SRCDIR}/wmma_extension/include)
target_include_directories(tsqr-tc PUBLIC ${SRCDIR}/hmma.f32.f32/include)
target_include_directories(tsqr-tc PUBLIC ${SRCDIR}/gemm_core_cuh/include)

# CUDA (2/2)
set_target_properties(tsqr-tc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Output
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIBDIR})

##########################################################################
# Tests
##########################################################################

# Directory
set(TESTSRCDIR test)

foreach(test batchqr tsqr qr256x256)
	set(test_out ${test}.test)
	add_executable(${test_out} ${TESTSRCDIR}/${test}.cu ${TESTSRCDIR}/utils.cu)
	target_include_directories(${test_out} PRIVATE ${INCDIR})
	target_include_directories(${test_out} PRIVATE ${SRCDIR}/cutf/include)
	target_link_libraries(${test_out} PRIVATE tsqr-tc)
	target_link_libraries(${test_out} PRIVATE
		CUDA::cublas
		CUDA::cusolver
		)
	#target_link_libraries(${test_out} PRIVATE -lculip_cublas)
	if (OpenMP_CXX_FOUND)
		target_link_libraries(${test_out} PRIVATE OpenMP::OpenMP_CXX)
	endif()
endforeach()
