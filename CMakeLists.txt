cmake_minimum_required(VERSION 2.6)
project (benchmark_phi)

set(CMAKE_CXX_COMPILER "icc")
set(CMAKE_BUILD_TYPE RELEASE)
#AVX bug: can't use -xavx; compiler doesn't generate aligned moves!
set(CMAKE_CXX_FLAGS_RELEASE "-openmp -vec-report=3 -ansi-alias -O3 -g")

set(alignment 64)

include_directories(include)
if(NOT DEFINED USE_DOUBLE)
  set(use_double -DUSE_DOUBLE=0)
else()
    set(use_double -DUSE_DOUBLE=${USE_DOUBLE})
endif()

add_definitions(-DALIGNMENT=${alignment} ${use_double})

add_executable(main src/main.cpp src/gemm.cpp)
