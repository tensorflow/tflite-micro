# Gemmlowp CMake file written for Debian.
# Copyright © 2016 Zhou Mo <cdluminate@gmail.com>
# Licence Apache-2.0

cmake_minimum_required(VERSION 3.7)

# Project
project(gemmlowp C CXX)

# Set C++11 as default standard
set(CMAKE_CXX_STANDARD 11)

get_filename_component(gemmlowp_src ${gemmlowp_SOURCE_DIR} PATH)

if(WIN32)
  # one can enable simd from the cmake command line, ie -DCMAKE_CXX_FLAGS="/arch:AVX2
  add_definitions(-DNOMINMAX -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI)
  add_definitions(/bigobj /nologo /EHsc /GF /MP /Gm- /wd4800 /wd4805 /wd4244)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # if we compile for windows with clang, allow inline asm
    add_definitions(-DGEMMLOWP_ALLOW_INLINE_ASM)
  endif()
else()
  set(EXTERNAL_LIBRARIES "pthread")
endif()

# Glob header files
file(GLOB gemmlowp_private_headers "${gemmlowp_src}/fixedpoint/*.h" "${gemmlowp_src}/internal/*.h")
file(GLOB gemmlowp_public_headers "${gemmlowp_src}/meta/*.h" "${gemmlowp_src}/public/*.h" "${gemmlowp_src}/profiling/*.h")
list(APPEND gemmlowp_headers ${gemmlowp_private_headers} ${gemmlowp_public_headers})

file(GLOB eight_bit_int_gemm_headers "${gemmlowp_src}/eight_bit_int_gemm/*.h")
list(APPEND eight_bit_int_gemm_public_headers ${eight_bit_int_gemm_headers} ${gemmlowp_public_headers})
file(GLOB eight_bit_int_gemm_sources_with_no_headers "${gemmlowp_src}/eight_bit_int_gemm/*.cc")

list(APPEND eight_bit_int_gemm_sources
            ${eight_bit_int_gemm_headers}
            ${eight_bit_int_gemm_sources_with_no_headers}
            ${gemmlowp_headers})

file(GLOB gemmlowp_test_headers "${gemmlowp_src}/test/*.h")
list(APPEND gemmlowp_test_headers ${gemmlowp_headers})

file(GLOB fixedpoint_private_headers "${gemmlowp_src}/fixedpoint/*.h")
list(APPEND fixedpoint_private_headers "${gemmlowp_src}/internal/common.h")

# Eight bit int gemm library
if(WIN32)
    add_library(eight_bit_int_gemm STATIC ${eight_bit_int_gemm_sources_with_no_headers})
else()
    add_library(eight_bit_int_gemm SHARED ${eight_bit_int_gemm_sources_with_no_headers})
endif()
target_link_libraries(eight_bit_int_gemm ${EXTERNAL_LIBRARIES})

# Benchmarks
add_executable(benchmark
    "${gemmlowp_src}/test/benchmark.cc" ${gemmlowp_test_headers})
target_link_libraries(benchmark ${EXTERNAL_LIBRARIES})

add_executable(benchmark_all_sizes
    "${gemmlowp_src}/test/benchmark_all_sizes.cc" ${gemmlowp_test_headers})
target_compile_options(benchmark_all_sizes PRIVATE -DBENCHMARK_8bit -DBENCHMARK_QUICK)
target_link_libraries(benchmark_all_sizes ${EXTERNAL_LIBRARIES})

# Gemmlowp test
add_executable(test_gemmlowp
    "${gemmlowp_src}/test/test.cc" "${gemmlowp_src}/test/test_data.cc" ${gemmlowp_test_headers})
target_link_libraries(test_gemmlowp eight_bit_int_gemm)

# Math helpers test
add_executable(test_math_helpers
    "${gemmlowp_src}/test/test_math_helpers.cc" ${gemmlowp_test_headers})

# BlockingCounter test
add_executable(test_blocking_counter
    "${gemmlowp_src}/test/test_blocking_counter.cc" ${gemmlowp_test_headers})
target_link_libraries(test_blocking_counter ${EXTERNAL_LIBRARIES})

# Allocator test
add_executable(test_allocator
    "${gemmlowp_src}/test/test_allocator.cc" ${gemmlowp_test_headers})

# FixedPoint test
add_executable(test_fixedpoint
    "${gemmlowp_src}/test/test_fixedpoint.cc" ${gemmlowp_test_headers})

# Add tests
enable_testing()
foreach(testname "test_math_helpers" "test_blocking_counter" "test_allocator" "test_fixedpoint" "test_gemmlowp")
  add_test(NAME ${testname} COMMAND "${testname}")
endforeach(testname)
