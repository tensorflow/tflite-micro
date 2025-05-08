// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
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
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "multi_thread_gemm.h"
#include "quantized_mul_kernels.h"
#include "single_thread_gemm.h"
#include "streams.h"

#define LHS_OFFSET (-127)
#define RHS_OFFSET (-127)
#define SUM_OFFSET (127)
#define MUL_OFFSET (1)
#define SHIFT (7)
#define FLOAT_SCALE (0.333f)

using namespace gemmlowp::meta;

// Input, output & kernel setups.

typedef GemmParams<std::uint8_t, std::uint8_t, RowMajorWithSum, ColumnMajorWithSum,
                   QuantizedStaticPreprocessed, RowMajor>
    ParamsColumnMajor;

typedef GemmParams<std::uint8_t, std::uint8_t, RowMajorWithSum, RowMajorWithSum,
                   QuantizedStaticPreprocessed, RowMajor>
    ParamsRowMajor;

typedef GemmParams<std::uint8_t, float, RowMajorWithSum, ColumnMajorWithSum,
                   QuantizedStaticPreprocessedAsFloat, RowMajor>
    ParamsColumnMajorAsFloat;

typedef GemmParams<std::uint8_t, float, RowMajorWithSum, RowMajorWithSum,
                   QuantizedStaticPreprocessedAsFloat, RowMajor>
    ParamsRowMajorAsFloat;

typedef GemmParams<std::uint8_t, std::int32_t, RowMajorWithSum, ColumnMajorWithSum,
                   QuantizedStaticPreprocessedAsInt32, RowMajor>
    ParamsColumnMajorAsInt32;

typedef GemmParams<std::uint8_t, std::int32_t, RowMajorWithSum, RowMajorWithSum,
                   QuantizedStaticPreprocessedAsInt32, RowMajor>
    ParamsRowMajorAsInt32;

typedef gemmlowp::WorkersPool Pool;
typedef SimpleContext<gemmlowp::WorkersPool> Context;

#ifdef LHS_PACK
typedef GemmExecutorPackLHSCacheFriendly<> Executor;
#else
typedef GemmExecutorPackRHSCacheFriendly<> Executor;
#endif

// Testing helper functions.

void prepare_test_data(std::uint8_t* data, std::int32_t rows, std::int32_t cols,
                       std::int32_t seed, std::int32_t seed_2) {
  std::int32_t value = seed;
  for (int i = 0; i < rows * cols; ++i) {
    data[i] = static_cast<std::uint8_t>(value);
    value = ((value * seed_2) + seed) % 256;
  }
}

template <typename CLEAR_TYPE>
void clear(int rows, int cols, CLEAR_TYPE* data) {
  for (int i = 0; i < rows * cols; ++i) {
    data[i] = 0;
  }
}

bool check_row_row(std::uint8_t* lhs, std::uint8_t* rhs, std::uint8_t* results, int rows,
                   int cols, int depth) {
  int wrong = 0;
  int rounding = (1 << (SHIFT - 1));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected += (static_cast<int>(lhs[depth * i + k]) + LHS_OFFSET) *
                    (static_cast<int>(rhs[depth * j + k]) + RHS_OFFSET);
      }
      expected += SUM_OFFSET * depth;
      expected *= MUL_OFFSET;
      expected += rounding;
      expected = (expected >> SHIFT);
      if (expected < 0) {
        expected = 0;
      } else if (expected > 255) {
        expected = 255;
      }
      expected = static_cast<int>(static_cast<std::uint8_t>(expected));
      int actual = static_cast<int>(results[i * cols + j]);
      if (actual != expected) {
        std::cout << "Wrong @" << i << "x" << j << " : " << actual
                  << " != " << expected << std::endl;
        wrong++;
      }
    }
  }
  if (wrong != 0) {
    std::cout << wrong << "/" << (rows * cols) << std::endl;
  }
  return wrong == 0;
}

bool check_row_col(std::uint8_t* lhs, std::uint8_t* rhs, std::uint8_t* results, int rows,
                   int cols, int depth) {
  int wrong = 0;
  int rounding = (1 << (SHIFT - 1));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected += (static_cast<int>(lhs[depth * i + k]) + LHS_OFFSET) *
                    (static_cast<int>(rhs[j + k * cols]) + RHS_OFFSET);
      }
      expected += SUM_OFFSET * depth;
      expected *= MUL_OFFSET;
      expected += rounding;
      expected = (expected >> SHIFT);
      if (expected < 0) {
        expected = 0;
      } else if (expected > 255) {
        expected = 255;
      }
      expected = static_cast<int>(static_cast<std::uint8_t>(expected));
      int actual = static_cast<int>(results[i * cols + j]);
      if (actual != expected) {
        wrong++;
      }
    }
  }
  return wrong == 0;
}

bool check_row_row_f(std::uint8_t* lhs, std::uint8_t* rhs, float* results, int rows,
                     int cols, int depth) {
  int wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected += (static_cast<int>(lhs[depth * i + k]) + LHS_OFFSET) *
                    (static_cast<int>(rhs[depth * j + k]) + RHS_OFFSET);
      }
      float expected_float = static_cast<float>(expected) * FLOAT_SCALE;
      float actual = results[i * cols + j];
      if (actual != expected_float) {
        wrong++;
      }
    }
  }
  return wrong == 0;
}

bool check_row_col_f(std::uint8_t* lhs, std::uint8_t* rhs, float* results, int rows,
                     int cols, int depth) {
  int wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected += (static_cast<int>(lhs[depth * i + k]) + LHS_OFFSET) *
                    (static_cast<int>(rhs[j + k * cols]) + RHS_OFFSET);
      }
      float expected_float = static_cast<float>(expected) * FLOAT_SCALE;
      float actual = results[i * cols + j];
      if (actual != expected_float) {
        wrong++;
      }
    }
  }
  return wrong == 0;
}

bool check_row_row_i32(std::uint8_t* lhs, std::uint8_t* rhs, std::int32_t* results, int rows,
                       int cols, int depth) {
  int wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected += (static_cast<int>(lhs[depth * i + k]) + LHS_OFFSET) *
                    (static_cast<int>(rhs[depth * j + k]) + RHS_OFFSET);
      }
      int actual = results[i * cols + j];
      if (actual != expected) {
        wrong++;
      }
    }
  }
  return wrong == 0;
}

bool check_row_col_i32(std::uint8_t* lhs, std::uint8_t* rhs, std::int32_t* results, int rows,
                       int cols, int depth) {
  int wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected += (static_cast<int>(lhs[depth * i + k]) + LHS_OFFSET) *
                    (static_cast<int>(rhs[j + k * cols]) + RHS_OFFSET);
      }
      int actual = results[i * cols + j];
      if (actual != expected) {
        wrong++;
      }
    }
  }
  return wrong == 0;
}

template <typename PARAMS, typename RESULT_TYPE>
void setup_params(std::uint8_t* lhs, std::uint8_t* rhs, RESULT_TYPE* result,
                  std::uint8_t* scratch, PARAMS* params) {
  params->lhs = lhs;
  params->rhs = rhs;
  params->result = result;
  params->scratch = scratch;

  params->left_stream.multiplicative_sum_offset = RHS_OFFSET;
  params->left_stream.additive_sum_offset = 0;

  params->right_stream.multiplicative_sum_offset = LHS_OFFSET;
  params->right_stream.additive_sum_offset = 0;
}

void setup_row_row(int m, int n, int k, ParamsRowMajor* params) {
  params->m = m;
  params->n = n;
  params->k = k;
  params->left_stream.count = k;
  params->left_stream.stride = k;
  params->left_stream.additive_sum_offset =
      SUM_OFFSET * k + k * LHS_OFFSET * RHS_OFFSET;
  params->right_stream.count = k;
  params->right_stream.stride = k;
  params->fused_kernel.kernel.count = k;
  params->fused_kernel.kernel.multiplicative_offset = MUL_OFFSET;
  params->fused_kernel.kernel.rounding_offset = (1 << (SHIFT - 1));
  params->fused_kernel.kernel.shift = -SHIFT;
  params->fused_kernel.output_stream.stride = n;
}

void setup_row_col(int m, int n, int k, ParamsColumnMajor* params) {
  params->m = m;
  params->n = n;
  params->k = k;
  params->left_stream.count = k;
  params->left_stream.stride = k;
  params->left_stream.additive_sum_offset =
      SUM_OFFSET * k + k * LHS_OFFSET * RHS_OFFSET;
  params->right_stream.count = k;
  params->right_stream.stride = n;
  params->fused_kernel.kernel.count = k;
  params->fused_kernel.kernel.multiplicative_offset = MUL_OFFSET;
  params->fused_kernel.kernel.rounding_offset = (1 << (SHIFT - 1));
  params->fused_kernel.kernel.shift = -SHIFT;
  params->fused_kernel.output_stream.stride = n;
}

void setup_row_row_f(int m, int n, int k, ParamsRowMajorAsFloat* params) {
  params->m = m;
  params->n = n;
  params->k = k;
  params->left_stream.count = k;
  params->left_stream.stride = k;
  params->left_stream.additive_sum_offset = k * LHS_OFFSET * RHS_OFFSET;
  params->right_stream.count = k;
  params->right_stream.stride = k;
  params->fused_kernel.kernel.count = k;
  params->fused_kernel.kernel.scale = FLOAT_SCALE;
  params->fused_kernel.output_stream.stride = n * sizeof(float);
}

void setup_row_col_f(int m, int n, int k, ParamsColumnMajorAsFloat* params) {
  params->m = m;
  params->n = n;
  params->k = k;
  params->left_stream.count = k;
  params->left_stream.stride = k;
  params->left_stream.additive_sum_offset = k * LHS_OFFSET * RHS_OFFSET;
  params->right_stream.count = k;
  params->right_stream.stride = n;
  params->fused_kernel.kernel.count = k;
  params->fused_kernel.kernel.scale = FLOAT_SCALE;
  params->fused_kernel.output_stream.stride = n * sizeof(float);
}

void setup_row_row_i32(int m, int n, int k, ParamsRowMajorAsInt32* params) {
  params->m = m;
  params->n = n;
  params->k = k;
  params->left_stream.count = k;
  params->left_stream.stride = k;
  params->left_stream.additive_sum_offset = k * LHS_OFFSET * RHS_OFFSET;
  params->right_stream.count = k;
  params->right_stream.stride = k;
  params->fused_kernel.kernel.count = k;
  params->fused_kernel.output_stream.stride = n * sizeof(std::int32_t);
}

void setup_row_col_i32(int m, int n, int k, ParamsColumnMajorAsInt32* params) {
  params->m = m;
  params->n = n;
  params->k = k;
  params->left_stream.count = k;
  params->left_stream.stride = k;
  params->left_stream.additive_sum_offset = k * LHS_OFFSET * RHS_OFFSET;
  params->right_stream.count = k;
  params->right_stream.stride = n;
  params->fused_kernel.kernel.count = k;
  params->fused_kernel.output_stream.stride = n * sizeof(std::int32_t);
}

int main() {
  ParamsRowMajor params_row;
  ParamsColumnMajor params_col;
  ParamsRowMajorAsFloat params_row_f;
  ParamsColumnMajorAsFloat params_col_f;
  ParamsRowMajorAsInt32 params_row_i32;
  ParamsColumnMajorAsInt32 params_col_i32;

  std::unique_ptr<std::uint8_t> lhs(new std::uint8_t[1024 * 1024]);
  std::unique_ptr<std::uint8_t> rhs(new std::uint8_t[1024 * 1024]);
  std::unique_ptr<std::uint8_t> result(new std::uint8_t[1024 * 1024]);
  std::unique_ptr<float> result_f(new float[1024 * 1024]);
  std::unique_ptr<std::int32_t> result_i32(new std::int32_t[1024 * 1024]);
  std::unique_ptr<std::uint8_t> scratch(new std::uint8_t[4048 * 1024]);

  setup_params(lhs.get(), rhs.get(), result.get(), scratch.get(), &params_row);
  setup_params(lhs.get(), rhs.get(), result.get(), scratch.get(), &params_col);
  setup_params(lhs.get(), rhs.get(), result_f.get(), scratch.get(),
               &params_row_f);
  setup_params(lhs.get(), rhs.get(), result_f.get(), scratch.get(),
               &params_col_f);
  setup_params(lhs.get(), rhs.get(), result_i32.get(), scratch.get(),
               &params_row_i32);
  setup_params(lhs.get(), rhs.get(), result_i32.get(), scratch.get(),
               &params_col_i32);

  Pool pool;
  Context context(4, &pool);

  for (int i = 1; i < 16; ++i) {
    for (int j = 1; j < 16; ++j) {
      for (int k = 1; k < 24; ++k) {
        prepare_test_data(lhs.get(), i, k, 11, 13);
        prepare_test_data(rhs.get(), j, k, 13, 17);

        clear(i, j, result.get());
        setup_row_row(i, j, k, &params_row);
        Gemm<Executor, ParamsRowMajor, 2, 4, 8>(params_row);
        if (!check_row_row(lhs.get(), rhs.get(), result.get(), i, j, k)) {
          std::cout << "Row: " << i << "x" << j << "x" << k << " : ERROR"
                    << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result.get());
        setup_row_col(i, j, k, &params_col);
        Gemm<Executor, ParamsColumnMajor, 2, 4, 8>(params_col);
        if (!check_row_col(lhs.get(), rhs.get(), result.get(), i, j, k)) {
          std::cout << "Column: " << i << "x" << j << "x" << k << " : ERROR"
                    << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_f.get());
        setup_row_row_f(i, j, k, &params_row_f);
        Gemm<Executor, ParamsRowMajorAsFloat, 2, 4, 8>(params_row_f);
        if (!check_row_row_f(lhs.get(), rhs.get(), result_f.get(), i, j, k)) {
          std::cout << "RowAsFloat: " << i << "x" << j << "x" << k << " : ERROR"
                    << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_f.get());
        setup_row_col_f(i, j, k, &params_col_f);
        Gemm<Executor, ParamsColumnMajorAsFloat, 2, 4, 8>(params_col_f);
        if (!check_row_col_f(lhs.get(), rhs.get(), result_f.get(), i, j, k)) {
          std::cout << "ColumnAsFloat: " << i << "x" << j << "x" << k
                    << " : ERROR" << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_i32.get());
        setup_row_row_i32(i, j, k, &params_row_i32);
        Gemm<Executor, ParamsRowMajorAsInt32, 2, 4, 8>(params_row_i32);
        if (!check_row_row_i32(lhs.get(), rhs.get(), result_i32.get(), i, j,
                               k)) {
          std::cout << "RowAsInt32: " << i << "x" << j << "x" << k << " : ERROR"
                    << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_i32.get());
        setup_row_col_i32(i, j, k, &params_col_i32);
        Gemm<Executor, ParamsColumnMajorAsInt32, 2, 4, 8>(params_col_i32);
        if (!check_row_col_i32(lhs.get(), rhs.get(), result_i32.get(), i, j,
                               k)) {
          std::cout << "ColumnAsInt32: " << i << "x" << j << "x" << k
                    << " : ERROR" << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }
      }
    }
  }

  for (int i = 1; i < 1024; i += 211) {
    for (int j = 1; j < 1024; j += 211) {
      for (int k = 8; k < 1024; k += 111) {
        prepare_test_data(lhs.get(), i, k, 11, 13);
        prepare_test_data(rhs.get(), j, k, 13, 17);

        clear(i, j, result.get());
        setup_row_row(i, j, k, &params_row);
        MultiThreadGemm<Context, Executor, ParamsRowMajor, 2, 4, 8>(&context,
                                                                    params_row);
        if (!check_row_row(lhs.get(), rhs.get(), result.get(), i, j, k)) {
          std::cout << "Row(MT): " << i << "x" << j << "x" << k << " : ERROR"
                    << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result.get());
        setup_row_col(i, j, k, &params_col);
        MultiThreadGemm<Context, Executor, ParamsColumnMajor, 2, 4, 8>(
            &context, params_col);
        if (!check_row_col(lhs.get(), rhs.get(), result.get(), i, j, k)) {
          std::cout << "Column(MT): " << i << "x" << j << "x" << k << " : ERROR"
                    << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_f.get());
        setup_row_row_f(i, j, k, &params_row_f);
        MultiThreadGemm<Context, Executor, ParamsRowMajorAsFloat, 2, 4, 8>(
            &context, params_row_f);
        if (!check_row_row_f(lhs.get(), rhs.get(), result_f.get(), i, j, k)) {
          std::cout << "RowAsFloat(MT): " << i << "x" << j << "x" << k
                    << " : ERROR" << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_f.get());
        setup_row_col_f(i, j, k, &params_col_f);
        MultiThreadGemm<Context, Executor, ParamsColumnMajorAsFloat, 2, 4, 8>(
            &context, params_col_f);
        if (!check_row_col_f(lhs.get(), rhs.get(), result_f.get(), i, j, k)) {
          std::cout << "ColumnAsFloat(MT): " << i << "x" << j << "x" << k
                    << " : ERROR" << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_i32.get());
        setup_row_row_i32(i, j, k, &params_row_i32);
        MultiThreadGemm<Context, Executor, ParamsRowMajorAsInt32, 2, 4, 8>(
            &context, params_row_i32);
        if (!check_row_row_i32(lhs.get(), rhs.get(), result_i32.get(), i, j,
                               k)) {
          std::cout << "RowAsInt32(MT): " << i << "x" << j << "x" << k
                    << " : ERROR" << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }

        clear(i, j, result_i32.get());
        setup_row_col_i32(i, j, k, &params_col_i32);
        MultiThreadGemm<Context, Executor, ParamsColumnMajorAsInt32, 2, 4, 8>(
            &context, params_col_i32);
        if (!check_row_col_i32(lhs.get(), rhs.get(), result_i32.get(), i, j,
                               k)) {
          std::cout << "ColumnAsInt32(MT): " << i << "x" << j << "x" << k
                    << " : ERROR" << std::endl;
          std::cout << "Exiting." << std::endl;
          std::exit(1);
        }
      }
    }
  }

  std::cout << "OK." << std::endl;
  return 0;
}
