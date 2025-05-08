// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unistd.h>
#ifdef __APPLE__
#include <sys/time.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <vector>

#include "../meta/legacy_multi_thread_gemm.h"
#include "../public/gemmlowp.h"
#include "test.h"
// lets include these so we make sure they always compile
#include "../meta/multi_thread_gemm.h"
#include "../meta/multi_thread_transform.h"
#include "../meta/legacy_multi_thread_common.h"

#if defined(__arm__) && !defined(GEMMLOWP_NEON)
#warning "Building without NEON support on ARM, check your compiler setup!"
#endif

double time() {
#ifdef __APPLE__
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + 1e-6 * t.tv_usec;
#else
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

void prepare_test_data(std::uint8_t* data, std::int32_t rows, std::int32_t cols,
                       std::int32_t seed, std::int32_t seed_2) {
  std::int32_t value = seed;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      data[i * cols + j] = static_cast<std::uint8_t>(value);
      value = ((value * seed_2) + seed) % 256;
    }
  }
}

void check_result(std::uint8_t* left, std::uint8_t* right, std::uint8_t* result,
                  std::int32_t rows, std::int32_t cols, std::int32_t depth,
                  std::int32_t lhs_offset, std::int32_t rhs_offset,
                  std::int32_t sum_offset, std::int32_t mul_offset,
                  std::int32_t shift) {
  std::int32_t rounding = (1 << (shift - 1));
  std::int32_t wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::int32_t expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected +=
            (static_cast<std::int32_t>(left[depth * i + k]) + lhs_offset) *
            (static_cast<std::int32_t>(right[depth * j + k]) + rhs_offset);
      }
      expected += sum_offset;
      expected *= mul_offset;
      expected += rounding;
      expected = (expected >> shift);
      if (expected < 0) {
        expected = 0;
      } else if (expected > 255) {
        expected = 255;
      }
      expected = static_cast<std::int32_t>(static_cast<std::uint8_t>(expected));
      std::int32_t actual = static_cast<std::int32_t>(result[i * cols + j]);
      if (actual != expected) {
        std::cout << "(" << i << ", " << j << "): " << expected << "!="
                  << actual << std::endl;
        wrong++;
      }
    }
  }
  if (wrong > 0) {
    std::cout << "Wrong: " << rows << "x" << cols << "x" << depth << " : "
              << wrong << "/" << (rows * cols) << std::endl
              << std::flush;
    std::exit(1);
  } else {
    std::cout << "." << std::flush;
  }
}

void check_result_f(std::uint8_t* left, std::uint8_t* right, float* result,
                    std::int32_t rows, std::int32_t cols, std::int32_t depth,
                    std::int32_t lhs_offset, std::int32_t rhs_offset,
                    float result_offset) {
  std::int32_t wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::int32_t expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected +=
            (static_cast<std::int32_t>(left[depth * i + k]) + lhs_offset) *
            (static_cast<std::int32_t>(right[depth * j + k]) + rhs_offset);
      }
      float expected_float = static_cast<float>(expected) * result_offset;
      float actual_float = result[i * cols + j];
      if (actual_float != expected_float) {
        std::cout << "(" << i << ", " << j << "): " << expected_float << "!="
                  << actual_float << std::endl;
        wrong++;
      }
    }
  }
  if (wrong > 0) {
    std::cout << "Wrong: " << rows << "x" << cols << "x" << depth << " : "
              << wrong << "/" << (rows * cols) << std::endl
              << std::flush;
    std::exit(1);
  } else {
    std::cout << "." << std::flush;
  }
}


void check_result_i32(std::uint8_t* left, std::uint8_t* right,
                      std::int32_t* result, std::int32_t rows,
                      std::int32_t cols, std::int32_t depth,
                      std::int32_t lhs_offset, std::int32_t rhs_offset) {
  std::int32_t wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::int32_t expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected +=
            (static_cast<std::int32_t>(left[depth * i + k]) + lhs_offset) *
            (static_cast<std::int32_t>(right[depth * j + k]) + rhs_offset);
      }
      std::int32_t actual = result[i * cols + j];
      if (actual != expected) {
        std::cout << "(" << i << ", " << j << "): " << expected << "!="
                  << actual << std::endl;
        wrong++;
      }
    }
  }
  if (wrong > 0) {
    std::cout << "Wrong: " << rows << "x" << cols << "x" << depth << " : "
              << wrong << "/" << (rows * cols) << std::endl
              << std::flush;
    std::exit(1);
  } else {
    std::cout << "." << std::flush;
  }
}

template <typename T>
void clear(T* result, std::int32_t rows, std::int32_t cols) {
  for (int i = 0; i < rows * cols; ++i) {
    result[i] = static_cast<T>(0);
  }
}

void test(std::uint8_t* scratch, std::uint8_t* lhs, std::uint8_t* rhs,
          std::int32_t m, std::int32_t n, std::int32_t k, std::uint8_t* result,
          gemmlowp::WorkersPool* pool, std::int32_t pool_size) {
  prepare_test_data(lhs, m, k, 11, 13);
  prepare_test_data(rhs, n, k, 177, 19);

  clear(result, m, n);
  gemmlowp::meta::multi_thread_gemm_q8(pool, pool_size, scratch, lhs, rhs, m, n,
                                       k, -127, -127, 127 * k, 1, 7, result);
  check_result(lhs, rhs, result, m, n, k, -127, -127, 127 * k, 1, 7);
}

void test_f(std::uint8_t* scratch, std::uint8_t* lhs, std::uint8_t* rhs,
            std::int32_t m, std::int32_t n, std::int32_t k, float* result,
            gemmlowp::WorkersPool* pool, std::int32_t pool_size) {
  prepare_test_data(lhs, m, k, 11, 13);
  prepare_test_data(rhs, n, k, 177, 19);

  clear(result, m, n);
  float scale = 1.0f / 1234567.8f;
  gemmlowp::meta::multi_thread_gemm_f(pool, pool_size, scratch, lhs, rhs, m, n,
                                      k, -127, -127, scale, result);
  check_result_f(lhs, rhs, result, m, n, k, -127, -127, scale);
}

void test_i32(std::uint8_t* scratch, std::uint8_t* lhs, std::uint8_t* rhs,
              std::int32_t m, std::int32_t n, std::int32_t k,
              std::int32_t* result, gemmlowp::WorkersPool* pool,
              std::int32_t pool_size) {
  prepare_test_data(lhs, m, k, 11, 13);
  prepare_test_data(rhs, n, k, 177, 19);

  clear(result, m, n);
  gemmlowp::meta::multi_thread_gemm_i32(pool, pool_size, scratch, lhs, rhs, m,
                                        n, k, -127, -127, result);
  check_result_i32(lhs, rhs, result, m, n, k, -127, -127);
}

void q_suite(int mi, int ni, int ki, int mx, int nx, int kx, int md, int nd,
             int kd, std::uint8_t* scratch, std::uint8_t* left,
             std::uint8_t* right, std::uint8_t* result,
             gemmlowp::WorkersPool* pool, int t) {
  for (int m = mi; m < mx; m += md) {
    for (int n = ni; n < nx; n += nd) {
      for (int k = ki; k < kx; k += kd) {
        test(scratch, left, right, m, n, k, result, pool, t);
      }
    }
  }
  std::cout << std::endl;
}

void f_suite(int mi, int ni, int ki, int mx, int nx, int kx, int md, int nd,
             int kd, std::uint8_t* scratch, std::uint8_t* left,
             std::uint8_t* right, float* result, gemmlowp::WorkersPool* pool,
             int t) {
  for (int m = mi; m < mx; m += md) {
    for (int n = ni; n < nx; n += nd) {
      for (int k = ki; k < kx; k += kd) {
        test_f(scratch, left, right, m, n, k, result, pool, t);
      }
    }
  }
  std::cout << std::endl;
}

void i32_suite(int mi, int ni, int ki, int mx, int nx, int kx, int md, int nd,
               int kd, std::uint8_t* scratch, std::uint8_t* left,
               std::uint8_t* right, std::int32_t* result,
               gemmlowp::WorkersPool* pool, int t) {
  for (int m = mi; m < mx; m += md) {
    for (int n = ni; n < nx; n += nd) {
      for (int k = ki; k < kx; k += kd) {
        test_i32(scratch, left, right, m, n, k, result, pool, t);
      }
    }
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  bool run_long_test = false;

  if (argc > 1 && strcmp(argv[1], "long")) {
    run_long_test = true;
  }

  const std::int32_t min_n = 1;
  const std::int32_t min_m = 1;
  const std::int32_t min_k = 8;

  const std::int32_t max_n = 1024;
  const std::int32_t max_m = 1024;
  const std::int32_t max_k = 2048;

  std::uint8_t* left = new std::uint8_t[max_m * max_k];
  std::uint8_t* right = new std::uint8_t[max_n * max_k];
  std::uint8_t* result = new std::uint8_t[max_m * max_n];
  float* result_float = new float[max_m * max_n];
  std::int32_t* result_i32 = new std::int32_t[max_m * max_n];
  std::uint8_t* scratch = new std::uint8_t[1024 * 1024 * 64];

  gemmlowp::WorkersPool pool;

  int max_repetitions = run_long_test ? 10 : 1;

  for (int repetitions = 0; repetitions < max_repetitions; ++repetitions) {
    int t = std::min(repetitions + 1, 4);
    std::cout << "Threads: " << t << std::endl << std::flush;

    std::cout << "Quantized 8 bit." << std::endl << std::flush;

    std::cout << "Small." << std::endl << std::flush;
    q_suite(1, 1, 1, 16, 16, 32, 1, 1, 1, scratch, left, right, result, &pool,
            t);

    if (run_long_test) {
      std::cout << "Big." << std::endl << std::flush;
      q_suite(1, 1, 1, 512, 512, 2048, 111, 111, 111, scratch, left, right,
              result, &pool, t);
    }

    std::cout << "Gemv." << std::endl << std::flush;
    q_suite(1, 1, 1, 2, 512, 2048, 1, 111, 111, scratch, left, right, result,
            &pool, t);
    q_suite(1, 1, 1, 512, 2, 2048, 111, 1, 111, scratch, left, right, result,
            &pool, t);

    std::cout << std::endl << "Floats." << std::endl << std::flush;

    std::cout << "Small." << std::endl << std::flush;
    f_suite(1, 1, 1, 16, 16, 32, 1, 1, 1, scratch, left, right, result_float,
            &pool, t);

    if (run_long_test) {
      std::cout << "Big." << std::endl << std::flush;
      f_suite(1, 1, 1, 512, 512, 2048, 111, 111, 111, scratch, left, right,
              result_float, &pool, t);
    }

    std::cout << "Gemv." << std::endl << std::flush;
    f_suite(1, 1, 1, 2, 512, 2048, 1, 111, 111, scratch, left, right,
            result_float, &pool, t);
    f_suite(1, 1, 1, 512, 2, 2048, 111, 1, 111, scratch, left, right,
            result_float, &pool, t);

    std::cout << std::endl << "Int32." << std::endl << std::flush;

    std::cout << "Small." << std::endl << std::flush;
    i32_suite(1, 1, 1, 16, 16, 32, 1, 1, 1, scratch, left, right, result_i32,
              &pool, t);

    if (run_long_test) {
      std::cout << "Big." << std::endl << std::flush;
      i32_suite(1, 1, 1, 512, 512, 2048, 111, 111, 111, scratch, left, right,
                result_i32, &pool, t);
    }

    std::cout << "Gemv." << std::endl << std::flush;
    i32_suite(1, 1, 1, 2, 512, 2048, 1, 111, 111, scratch, left, right,
              result_i32, &pool, t);
    i32_suite(1, 1, 1, 512, 2, 2048, 111, 1, 111, scratch, left, right,
              result_i32, &pool, t);

    std::cout << std::endl << std::flush;
  }

  std::cout << "Done." << std::endl << std::flush;
}
