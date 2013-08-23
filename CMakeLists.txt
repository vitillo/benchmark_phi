cmake_minimum_required(VERSION 2.6)
project (benchmark_phi)

set(CMAKE_CXX_COMPILER "icc")
set(CMAKE_BUILD_TYPE RELEASE)
#AVX bug: can't use -xavx; compiler doesn't generate aligned moves!
#Furthermore OpenCL can't generate efficient AVX code... so compare only SSE to SSE
set(CMAKE_CXX_FLAGS_RELEASE "-openmp -lOpenCL -vec-report=1 -ansi-alias -O3 -g")

set(alignment 64)

include_directories(include)
if(NOT DEFINED USE_DOUBLE)
  set(use_double -DUSE_DOUBLE=0)
else()
    set(use_double -DUSE_DOUBLE=${USE_DOUBLE})
endif()

add_definitions(-DALIGNMENT=${alignment} ${use_double})

add_executable(icc src/icc.cpp src/gemm.cpp)
add_executable(ocl src/opencl.cpp)
