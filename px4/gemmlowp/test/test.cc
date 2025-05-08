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

#include "test.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "../eight_bit_int_gemm/eight_bit_int_gemm.h"
#include "../internal/kernel_reference.h"
#include "test_data.h"

namespace gemmlowp {

void ReferenceEightBitIntGemm(bool transpose_a, bool transpose_b,
                              bool transpose_c, int m, int n, int k,
                              const std::uint8_t* a, std::int32_t a_offset,
                              int lda, const std::uint8_t* b,
                              std::int32_t b_offset, int ldb, std::uint8_t* c,
                              std::int32_t c_offset, std::int32_t c_mult_int,
                              std::int32_t c_shift, int ldc) {
  ScopedProfilingLabel("ReferenceEightBitIntGemm");
  assert((c_shift >= 0) && (c_shift <= 32));

  assert(a != nullptr);
  assert(b != nullptr);
  assert(c != nullptr);

  int a_i_stride;
  int a_l_stride;
  if (transpose_a) {
    a_i_stride = lda;
    a_l_stride = 1;
  } else {
    a_i_stride = 1;
    a_l_stride = lda;
  }
  int b_j_stride;
  int b_l_stride;
  if (transpose_b) {
    b_j_stride = 1;
    b_l_stride = ldb;
  } else {
    b_j_stride = ldb;
    b_l_stride = 1;
  }
  int c_i_stride;
  int c_j_stride;
  if (transpose_c) {
    c_i_stride = ldc;
    c_j_stride = 1;
  } else {
    c_i_stride = 1;
    c_j_stride = ldc;
  }
  int i, j, l;

  const std::int32_t kRoundingTerm = (c_shift < 1) ? 0 : (1 << (c_shift - 1));

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      std::int32_t total = 0;
      for (l = 0; l < k; l++) {
        const int a_index = i * a_i_stride + l * a_l_stride;
        const std::uint8_t a_as_byte = a[a_index];
        const std::int32_t a_as_int =
            static_cast<std::int32_t>(a_as_byte) + a_offset;
        const int b_index = j * b_j_stride + l * b_l_stride;
        const std::uint8_t b_as_byte = b[b_index];
        const std::int32_t b_as_int =
            static_cast<std::int32_t>(b_as_byte) + b_offset;
        const std::int32_t mult_as_int = a_as_int * b_as_int;
        total += mult_as_int;
      }
      std::int32_t output =
          (((total + c_offset) * c_mult_int) + kRoundingTerm) >> c_shift;
      if (output > 255) {
        output = 255;
      }
      if (output < 0) {
        output = 0;
      }
      const int c_index = i * c_i_stride + j * c_j_stride;
      c[c_index] = static_cast<std::uint8_t>(output);
    }
  }
}

typedef VectorMap<const std::int32_t, VectorShape::Col> OffsetColMap;
typedef VectorMap<const std::int32_t, VectorShape::Row> OffsetRowMap;
typedef VectorDup<const std::int32_t, VectorShape::Col> OffsetColDup;
typedef VectorDup<const std::int32_t, VectorShape::Row> OffsetRowDup;

// *GemmWrapper's allow to wrap various Gemm functions in a uniform
// interface, so we can use the same testing code to test all of them

template <typename Kernel, typename Scalar, typename tBitDepthParams>
struct SingleThreadGemmWrapper {
  typedef tBitDepthParams BitDepthParams;

  static const char* Name() {
    static char buf[256];
    snprintf(buf, sizeof(buf), "SingleThreadGemm, Kernel: %s", Kernel().Name());
    return buf;
  }

  typedef SingleThreadGemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static bool Gemm(Context* context,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    ScopedProfilingLabel("SingleThreadGemmWrapper::Gemm");
    const int rows = lhs.rows();
    const int cols = rhs.cols();
    if (rows < cols) {
      // SingleThreadGemm is never called with rows < cols.
      // That case is handled earlier.
      return false;
    }
    const OffsetColDup lhs_offset_vector(lhs_offset, rows);
    const OffsetRowDup rhs_offset_vector(rhs_offset, cols);
    SingleThreadGemm<typename Kernel::Format, Scalar, Scalar, BitDepthParams,
                     LhsOrder, RhsOrder, ResultOrder, OffsetColDup,
                     OffsetRowDup>(
        context, Kernel(), lhs, rhs, result, lhs_offset_vector,
        rhs_offset_vector,
        MakeStandardOutputPipeline(result_offset, result_mult_int,
                                   result_shift));
    return true;
  }
};

template <typename Kernel, typename Scalar, typename tBitDepthParams>
struct MultiThreadGemmWrapper {
  typedef tBitDepthParams BitDepthParams;

  static const char* Name() {
    static char buf[256];
    snprintf(buf, sizeof(buf), "MultiThreadGemm, Kernel: %s", Kernel().Name());
    return buf;
  }

  typedef MultiThreadGemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static bool Gemm(Context* context,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    ScopedProfilingLabel("MultiThreadGemmWrapper::Gemm");
    context->set_max_num_threads(0);
    const int rows = lhs.rows();
    const int cols = rhs.cols();
    if (rows < cols) {
      // SingleThreadGemm is never called with rows < cols.
      // That case is handled earlier.
      return false;
    }
    const OffsetColDup lhs_offset_vector(lhs_offset, rows);
    const OffsetRowDup rhs_offset_vector(rhs_offset, cols);
    MultiThreadGemm<typename Kernel::Format, Scalar, Scalar, BitDepthParams,
                    LhsOrder, RhsOrder, ResultOrder, OffsetColDup,
                    OffsetRowDup>(
        context, Kernel(), lhs, rhs, result, lhs_offset_vector,
        rhs_offset_vector,
        MakeStandardOutputPipeline(result_offset, result_mult_int,
                                   result_shift));
    return true;
  }
};

template <typename Scalar, typename tBitDepthParams>
struct PublicGemmWrapper {
  typedef tBitDepthParams BitDepthParams;

  static const char* Name() { return "public Gemm"; }

  typedef GemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static bool Gemm(Context* context,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    ScopedProfilingLabel("PublicGemmWrapper::Gemm");
    gemmlowp::Gemm<std::uint8_t, BitDepthParams, LhsOrder, RhsOrder,
                   ResultOrder>(context, lhs, rhs, result, lhs_offset,
                                rhs_offset, result_offset, result_mult_int,
                                result_shift);
    return true;
  }
};

template <eight_bit_int_gemm::BitDepthSetting BitDepth>
struct BitDepthParamsForSettings {};

template <>
struct BitDepthParamsForSettings<eight_bit_int_gemm::BitDepthSetting::A8B8>
    : DefaultL8R8BitDepthParams {};

template <>
struct BitDepthParamsForSettings<eight_bit_int_gemm::BitDepthSetting::A5B7>
    : DefaultL7R5BitDepthParams {};

template <typename Scalar, eight_bit_int_gemm::BitDepthSetting BitDepth>
struct EightBitIntGemmWrapper {
  typedef BitDepthParamsForSettings<BitDepth> BitDepthParams;

  static const char* Name() { return "EightBitIntGemm"; }

  typedef void Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static bool Gemm(Context*, const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    ScopedProfilingLabel("EightBitIntGemmWrapper::Gemm");
    const bool transpose_c = ResultOrder == MapOrder::RowMajor;
    const bool transpose_a = LhsOrder == MapOrder::RowMajor;
    const bool transpose_b = RhsOrder == MapOrder::RowMajor;
    eight_bit_int_gemm::EightBitIntGemm(
        transpose_a, transpose_b, transpose_c, lhs.rows(), rhs.cols(),
        lhs.cols(), lhs.data(), lhs_offset, lhs.stride(), rhs.data(),
        rhs_offset, rhs.stride(), result->data(), result_offset,
        result_mult_int, result_shift, result->stride(), BitDepth);
    return true;
  }
};

template <typename Scalar>
struct ReferenceEightBitIntGemmWrapper {
  typedef DefaultL8R8BitDepthParams BitDepthParams;

  static const char* Name() { return "ReferenceEightBitIntGemm"; }

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static bool Gemm(bool transpose_a, bool transpose_b, bool transpose_c,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    ScopedProfilingLabel("ReferenceEightBitIntGemmWrapper::Gemm");
    ReferenceEightBitIntGemm(transpose_a, transpose_b, transpose_c, lhs.rows(),
                             rhs.cols(), lhs.cols(), lhs.data(), lhs_offset,
                             lhs.stride(), rhs.data(), rhs_offset, rhs.stride(),
                             result->data(), result_offset, result_mult_int,
                             result_shift, result->stride());
    return true;
  }
};

const char* OrderName(MapOrder order) {
  return order == MapOrder::ColMajor ? "ColMajor" : "RowMajor";
}

struct ResultStats {
  ResultStats()
      : count(0),
        med_val(0),
        mean_signed_diff(0),
        med_signed_diff(0),
        med_unsigned_diff(0),
        max_unsigned_diff(0) {}

  int count;
  int med_val;
  float mean_signed_diff;
  int med_signed_diff;
  int med_unsigned_diff;
  int max_unsigned_diff;

  std::vector<int> count_diff_by_pot_slice;
};

void GetResultStats(const std::uint8_t* actual, const std::uint8_t* expected,
                    size_t count, ResultStats* stats) {
  ScopedProfilingLabel("GetResultStats");
  std::vector<std::uint8_t> results;
  std::vector<std::int16_t> signed_diffs;
  std::vector<std::uint8_t> unsigned_diffs;
  std::int64_t signed_diffs_sum = 0;
  for (size_t i = 0; i < count; i++) {
    results.push_back(actual[i]);
    std::int16_t signed_diff = actual[i] - expected[i];
    signed_diffs.push_back(signed_diff);
    unsigned_diffs.push_back(std::abs(signed_diff));
    signed_diffs_sum += signed_diff;
  }

  std::sort(results.begin(), results.end());
  std::sort(signed_diffs.begin(), signed_diffs.end());
  std::sort(unsigned_diffs.begin(), unsigned_diffs.end());

  const size_t middle = count / 2;

  stats->count = count;
  stats->med_val = results[middle];
  stats->mean_signed_diff = float(signed_diffs_sum) / count;
  stats->med_signed_diff = signed_diffs[middle];
  stats->med_unsigned_diff = unsigned_diffs[middle];
  stats->max_unsigned_diff = unsigned_diffs.back();

  // Size 9 for 9 different POT values: 2^0, ..., 2^8
  stats->count_diff_by_pot_slice.resize(9);
  auto cur = unsigned_diffs.begin();
  size_t checksum = 0;
  for (int exponent = 0; exponent < 9; exponent++) {
    int pot = 1 << exponent;
    auto next = std::lower_bound(cur, unsigned_diffs.end(), pot);
    checksum += stats->count_diff_by_pot_slice[exponent] = next - cur;
    cur = next;
  }
  assert(checksum == count);
}

struct ResultStatsBounds {
  ResultStatsBounds()
      : mean_signed_diff(0),
        med_signed_diff(0),
        med_unsigned_diff(0),
        max_unsigned_diff(0) {}

  float mean_signed_diff;
  int med_signed_diff;
  int med_unsigned_diff;
  int max_unsigned_diff;
};

bool CheckResultStatsBounds(const ResultStats& stats,
                            const ResultStatsBounds& bounds) {
  return stats.max_unsigned_diff <= bounds.max_unsigned_diff &&
         stats.med_unsigned_diff <= bounds.med_unsigned_diff &&
         std::abs(stats.med_signed_diff) <= bounds.med_signed_diff &&
         std::abs(stats.mean_signed_diff) <= bounds.mean_signed_diff;
}

void ReportResultStats(const ResultStats& stats,
                       const ResultStatsBounds& bounds) {
  printf("    number of matrix entries: %d\n", stats.count);
  printf("    median value: %d\n", stats.med_val);
  printf("    median unsigned diff: %d (tolerating %d)\n",
         stats.med_unsigned_diff, bounds.med_unsigned_diff);
  printf("    max unsigned diff: %d (tolerating %d)\n", stats.max_unsigned_diff,
         bounds.max_unsigned_diff);
  printf("    median signed diff: %d (tolerating %d)\n", stats.med_signed_diff,
         bounds.med_signed_diff);
  printf("    mean signed diff: %.3g (tolerating %.3g)\n",
         stats.mean_signed_diff, bounds.mean_signed_diff);

  printf("No error: %.2f %% of entries\n",
         100.f * stats.count_diff_by_pot_slice[0] / stats.count);
  for (int exponent = 1; exponent < 9; exponent++) {
    printf("Error in %d..%d range: %.2f %% of entries\n", 1 << (exponent - 1),
           (1 << exponent) - 1,
           100.f * stats.count_diff_by_pot_slice[exponent] / stats.count);
  }
}

// Our approach to choosing result_shift values for testing, is bisection.
// This function takes an interval, [result_shift_min .. result_shift_max].
// If too much saturation occurred in either direction, it bisects accordingly,
// recursing until the interval contains only one value.
// The primary reason why we prefer this over computing optimal shift values,
// is that we actually want to exercise some saturation, as there is nontrivial
// code handling that in gemmlowp.
// Secondarily, this is faster than computing optimal shifts, since in 90% of
// cases the first-tried shift value 16 turns out to be good enough.
template <typename GemmWrapper, typename LhsType, typename RhsType,
          typename ResultType>
void test_gemm_impl(typename GemmWrapper::Context* context, const LhsType& lhs,
                    const RhsType& rhs, ResultType* result, int lhs_offset,
                    int rhs_offset, int result_offset, int result_mult_int,
                    int result_shift_min, int result_shift_max) {
  const int rows = lhs.rows();
  const int cols = rhs.cols();
  Check(lhs.cols() == rhs.rows());
  const int depth = lhs.cols();

  const int result_shift = (result_shift_min + result_shift_max) / 2;

  if (!GemmWrapper::Gemm(context, lhs.const_map(), rhs.const_map(),
                         &result->map(), lhs_offset, rhs_offset, result_offset,
                         result_mult_int, result_shift)) {
    // Internal GEMM functions are not required to handle all cases
    // (e.g. rows < cols) as these are supposed to have been handled
    // ahead of them. Their test wrappers return false in that case.
    return;
  }

  typedef typename ResultType::Scalar Scalar;
  static const MapOrder kLhsOrder = LhsType::kOrder;
  static const MapOrder kRhsOrder = RhsType::kOrder;
  static const MapOrder kResultOrder = ResultType::kOrder;
  ResultType ref_result(rows, cols);
  const bool transpose_c = kResultOrder == MapOrder::RowMajor;
  const bool transpose_a = kLhsOrder == MapOrder::RowMajor;
  const bool transpose_b = kRhsOrder == MapOrder::RowMajor;
  ReferenceEightBitIntGemmWrapper<Scalar>::Gemm(
      transpose_a, transpose_b, transpose_c, lhs.const_map(), rhs.const_map(),
      &ref_result.map(), lhs_offset, rhs_offset, result_offset, result_mult_int,
      result_shift);

  typedef typename GemmWrapper::BitDepthParams BitDepthParams;

  ResultStats stats;
  GetResultStats(result->data(), ref_result.data(), rows * cols, &stats);

  // Adjust shifts until we get meaningful results
  int new_result_shift_min = result_shift_min;
  int new_result_shift_max = result_shift_max;
  bool retry = false;

  if (stats.med_val < 32) {
    new_result_shift_max = (result_shift_min + result_shift_max) / 2;
    retry = true;
  }

  if (stats.med_val > 224) {
    new_result_shift_min = (result_shift_min + result_shift_max) / 2;
    retry = true;
  }

  if (retry) {
    if (result_shift_min != result_shift_max) {
      test_gemm_impl<GemmWrapper>(context, lhs, rhs, result, lhs_offset,
                                  rhs_offset, result_offset, result_mult_int,
                                  new_result_shift_min, new_result_shift_max);
    }
    return;
  }

  ResultStatsBounds bounds;

  // Check results
  const bool good = CheckResultStatsBounds(stats, bounds);

  printf(
      "%s: %dx%dx%d %s x %s -> %s, %s, offsets %d/%d/%d, mult %d, shift %d\n",
      good ? "PASS" : "FAIL", rows, depth, cols, OrderName(kLhsOrder),
      OrderName(kRhsOrder), OrderName(kResultOrder), GemmWrapper::Name(),
      lhs_offset, rhs_offset, result_offset, result_mult_int, result_shift);

  if (!good) {
    ReportResultStats(stats, bounds);

    int bad_coeffs_printed = 0;
    for (int c = 0; c < result->cols() && bad_coeffs_printed < 200; c++) {
      for (int r = 0; r < result->rows() && bad_coeffs_printed < 200; r++) {
        if (ref_result(r, c) != (*result)(r, c)) {
          printf("bad coeff: at (%d, %d), expected %d, got %d\n", r, c,
                 ref_result(r, c), (*result)(r, c));
          bad_coeffs_printed++;
        }
      }
    }
  }

  Check(good);
}

template <typename GemmWrapper, typename LhsType, typename RhsType,
          typename ResultType>
void test_gemm(typename GemmWrapper::Context* context, const LhsType& lhs,
               const RhsType& rhs, ResultType* result, int lhs_offset,
               int rhs_offset, int result_offset, int result_mult_int) {
  test_gemm_impl<GemmWrapper>(context, lhs, rhs, result, lhs_offset, rhs_offset,
                              result_offset, result_mult_int, 0, 32);
}

enum class WhatParamsToTest {
  All,
  OnlyGenericCase,
};

template <typename GemmWrapper, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder>
void test_gemm(typename GemmWrapper::Context* context, int rows, int depth,
               int cols, WhatParamsToTest params_to_test) {
  typedef std::uint8_t Scalar;
  typedef Matrix<Scalar, LhsOrder> LhsType;
  using BitDepthParams = typename GemmWrapper::BitDepthParams;
  LhsType lhs(rows, depth);
  MakeRandom<typename BitDepthParams::LhsRange>(&lhs);
  typedef Matrix<Scalar, RhsOrder> RhsType;
  RhsType rhs(depth, cols);
  MakeRandom<typename BitDepthParams::RhsRange>(&rhs);
  typedef Matrix<Scalar, ResultOrder> ResultType;
  ResultType result(rows, cols);
  MakeZero(&result);

  if (params_to_test == WhatParamsToTest::All) {
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 0, 0, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 10, 0, 0, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 10, 0, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 0, 10, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 0, 0, 10);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 10, 10, 10, 10);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 256, 1, 17, 4);
  }
  test_gemm<GemmWrapper>(context, lhs, rhs, &result, -75, -91, 74980, 123);
}

enum class WhatOrdersToTest { All, OnlyRCC };

template <typename GemmWrapper>
void test_gemm(typename GemmWrapper::Context* context, int rows, int depth,
               int cols, WhatParamsToTest params_to_test,
               WhatOrdersToTest orders_to_test) {
#define GEMMLOWP_ONE_TEST(LhsOrder, RhsOrder, ResultOrder)         \
  do {                                                             \
    test_gemm<GemmWrapper, MapOrder::LhsOrder, MapOrder::RhsOrder, \
              MapOrder::ResultOrder>(context, rows, depth, cols,   \
                                     params_to_test);              \
  } while (false)

  if (orders_to_test == WhatOrdersToTest::All) {
    GEMMLOWP_ONE_TEST(ColMajor, ColMajor, ColMajor);
    GEMMLOWP_ONE_TEST(RowMajor, ColMajor, ColMajor);
    GEMMLOWP_ONE_TEST(ColMajor, RowMajor, ColMajor);
    GEMMLOWP_ONE_TEST(RowMajor, RowMajor, ColMajor);

    GEMMLOWP_ONE_TEST(ColMajor, ColMajor, RowMajor);
    GEMMLOWP_ONE_TEST(RowMajor, ColMajor, RowMajor);
    GEMMLOWP_ONE_TEST(ColMajor, RowMajor, RowMajor);
    GEMMLOWP_ONE_TEST(RowMajor, RowMajor, RowMajor);
  } else {
    GEMMLOWP_ONE_TEST(RowMajor, ColMajor, ColMajor);
  }

#undef GEMMLOWP_ONE_TEST
}

template <typename Kernel>
void test_gemm_kernel(MultiThreadGemmContext* context) {
  typedef MultiThreadGemmWrapper<Kernel, std::uint8_t,
                                 DefaultL8R8BitDepthParams>
      GemmWrapper;
  test_gemm<GemmWrapper>(context, 1, 1, 1, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 2, 2, 2, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 3, 3, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4, 4, 4, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 5, 5, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 9, 11, 13, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 50, 50, 50, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 200, 200, 200,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 50, 5000, 50,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
}

template <typename GemmWrapper>
void test_gemm(typename GemmWrapper::Context* context) {
  test_gemm<GemmWrapper>(context, 1, 1, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 2, 1, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 2, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1, 2, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 2, 2, 2, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 3, 3, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4, 4, 4, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 5, 5, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 6, 6, 6, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 5, 7, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 7, 3, 5, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 7, 3, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 8, 8, 8, WhatParamsToTest::All,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 16, 16, 16, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 32, 32, 32, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 64, 64, 64, WhatParamsToTest::All,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 128, 128, 128, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  test_gemm<GemmWrapper>(context, 16, 17, 16, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 37, 55, 73, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 57, 87, 117, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 93, 83, 73, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 109, 89, 99, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 78, 101, 82, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  test_gemm<GemmWrapper>(context, 512, 512, 512,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1024, 1024, 1024,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 567, 2345, 123,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 100, 5000, 100,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1, 1000, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1000, 1, 1, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1000, 1, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1000, 1000,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1000, 1, 1000,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1000, 1000, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 777, 3456, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4567, 555, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);

  // Test all storage orders
  test_gemm<GemmWrapper>(context, 70, 90, 110, WhatParamsToTest::All,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 300, 400, 500,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::All);
}

template <typename GemmWrapper>
void test_gemv(typename GemmWrapper::Context* context) {
  test_gemm<GemmWrapper>(context, 2, 2, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 3, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4, 4, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 5, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 6, 6, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 5, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 7, 3, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 7, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 8, 8, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 32, 32, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 128, 128, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 321, 123, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  // Test all storage orders
  test_gemm<GemmWrapper>(context, 70, 90, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 300, 400, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::All);
}

const char* GetBitDepthName(eight_bit_int_gemm::BitDepthSetting b) {
  switch (b) {
    case eight_bit_int_gemm::BitDepthSetting::A8B8:
      return "Lhs: 8 bit, Rhs: 8 bit";
    case eight_bit_int_gemm::BitDepthSetting::A5B7:
      return "(legacy, no longer requantizing) Lhs: 7 bit, Rhs: 5 bit";
    default:
      abort();
      return nullptr;
  }
}

// Runs a small set of hand-picked data for per-channel quantized data.
// This test case comes from a set of 2 2x2 convolution filters run over a 3x3
// image.
void TestWithSmallDataPerChannelQuantization() {
  const int m = 2;
  const int n = 9;
  const int k = 12;

  // 12 x 2, columnwise.
  const std::uint8_t a_data[] = {0,  0,   0,   0,   0,  0,   0,   0,
                                 0,  255, 255, 255, 64, 64,  64,  64,
                                 64, 64,  0,   0,   0,  255, 255, 255};
  const int lda = k;
  int a_offset[] = {0, -64};
  MatrixMap<const std::uint8_t, MapOrder::RowMajor> lhs(a_data, m, k, lda);
  const OffsetColMap lhs_offset(a_offset, m);

  // 12 x 9, columnwise.
  const std::uint8_t b_data[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,
      0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   127,
      127, 127, 0,   0,   0,   127, 127, 127, 0,   0,   0,   255, 255, 255,
      0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   127, 127, 127, 0,   0,   0,   127,
      127, 127, 0,   0,   0,   0,   0,   0,   127, 127, 127, 127, 127, 127,
      0,   0,   0,   0,   0,   0,   127, 127, 127, 127, 127, 127, 0,   0,
      0,   127, 127, 127, 127, 127, 127, 127, 127, 127};
  const int ldb = k;
  int b_offset = -127;
  MatrixMap<const std::uint8_t, MapOrder::ColMajor> rhs(b_data, k, n, ldb);
  const OffsetRowDup rhs_offset(b_offset, rhs.cols());

  // 2 x 9, columnwise.
  const std::uint8_t expected_c_data[] = {255, 255, 0,   0,   127, 159,
                                          0,   64,  0,   64,  127, 159,
                                          127, 127, 127, 127, 127, 127};
  const int ldc = m;
  int c_offset[] = {97155, 97346};
  int c_mult_int[] = {2741, 2741};
  const int c_shift = 21;

  const int c_count = m * n;
  std::unique_ptr<std::uint8_t[]> output_data(new std::uint8_t[c_count]);
  MatrixMap<std::uint8_t, MapOrder::ColMajor> result(output_data.get(), m, n,
                                                     ldc);
  const OffsetColMap result_offset(c_offset, m);
  const OffsetColMap result_mult_int(c_mult_int, m);
  const int result_shift = c_shift;

  GemmContext gemm_context;
  auto output_pipeline = MakeStandardOutputPipeline<VectorShape::Col>(
      result_offset, result_mult_int, result_shift);
  GemmWithOutputPipelinePC<std::uint8_t, std::uint8_t,
                           DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, lhs_offset, rhs_offset,
      output_pipeline);

  ResultStats stats;
  GetResultStats(output_data.get(), expected_c_data, c_count, &stats);

  ResultStatsBounds bounds;
  const bool good = CheckResultStatsBounds(stats, bounds);
  printf("TestWithSmallDataPerChannelQuantization: %s\n",
         good ? "PASS" : "FAIL");
  ReportResultStats(stats, bounds);
  Check(good);
}

// Runs a larger set of hand-picked data for per-channel quantized data.
// This test case comes from a set of 22 3x3 convolution filters run over a 5x5
// image.  Right now, I have 7 different filters and 15 copies of the first
// filter to make sure NEON code path that processes 16 rows at a time is
// covered.
void TestWithLargeDataPerChannelQuantization() {
  const int m = 22;
  const int n = 25;
  const int k = 27;

  // 27 x 22, column-wise.
  const std::uint8_t a_data[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   127, 127, 127, 255, 255, 255, 127, 127, 127,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   127, 127, 127,
      0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,
      127, 127, 127, 0,   0,   0,   51,  51,  51,  51,  51,  51,  51,  51,  51,
      0,   0,   0,   255, 255, 255, 0,   0,   0,   51,  51,  51,  51,  51,  51,
      51,  51,  51,  51,  51,  51,  0,   0,   0,   51,  51,  51,  51,  51,  51,
      255, 255, 255, 51,  51,  51,  51,  51,  51,  0,   0,   0,   51,  51,  51,
      0,   0,   0,   64,  64,  64,  0,   0,   0,   64,  64,  64,  255, 255, 255,
      64,  64,  64,  0,   0,   0,   64,  64,  64,  0,   0,   0,   36,  36,  36,
      0,   0,   0,   36,  36,  36,  0,   0,   0,   255, 255, 255, 0,   0,   0,
      36,  36,  36,  0,   0,   0,   36,  36,  36,  0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,
  };
  const int lda = k;
  int a_offset[] = {0, 0, 0, -51, -51, 0, -36, 0, 0, 0, 0,
                    0, 0, 0, 0,   0,   0, 0,   0, 0, 0, 0};
  MatrixMap<const std::uint8_t, MapOrder::RowMajor> lhs(a_data, m, k, lda);
  const OffsetColMap lhs_offset(a_offset, m);

  // 27 x 25, column-wise.
  const std::uint8_t b_data[] = {
      127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 119, 119,
      119, 119, 119, 119, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127,
      127, 127, 127, 127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127,
      127, 127, 127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127, 127,
      127, 127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127, 127, 127,
      127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127, 127, 127,
      119, 119, 119, 119, 119, 119, 127, 127, 127, 127, 127, 127, 119, 119,
      119, 119, 119, 119, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127,
      127, 127, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 136, 136, 136, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      136, 136, 136, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 136, 136, 136, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127, 127,
      119, 119, 119, 119, 119, 119, 127, 127, 127, 119, 119, 119, 119, 119,
      119, 127, 127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127,
      127, 127, 119, 119, 119, 119, 119, 119, 127, 127, 127, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 136, 136, 136, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      136, 136, 136, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 136, 136, 136, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 127, 127, 127, 119, 119, 119, 119, 119,
      119, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127, 127, 127, 127,
      127, 127, 119, 119, 119, 119, 119, 119, 127, 127, 127, 119, 119, 119,
      119, 119, 119, 127, 127, 127, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 136, 136, 136, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      136, 136, 136, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 136, 136, 136, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127, 127, 127, 119,
      119, 119, 119, 119, 119, 127, 127, 127, 127, 127, 127, 119, 119, 119,
      119, 119, 119, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127, 127,
      127, 127, 127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127, 127,
      127, 127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127, 127, 127,
      127, 127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 119, 119, 119,
      119, 119, 119, 119, 119, 119, 119, 119, 119, 127, 127, 127, 127, 127,
      127, 127, 127, 127, 119, 119, 119, 119, 119, 119, 127, 127, 127, 119,
      119, 119, 119, 119, 119, 127, 127, 127, 127, 127, 127, 127, 127, 127,
      127, 127, 127};
  const int ldb = k;
  int b_offset = -127;
  MatrixMap<const std::uint8_t, MapOrder::ColMajor> rhs(b_data, k, n, ldb);
  const OffsetRowDup rhs_offset(b_offset, rhs.cols());

  // 22 x 25, column-wise.
  const std::uint8_t expected_c_data[] = {
      7,   37,  37,  67,  67,  39,  79,  7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   37,  87,  67,  23,  91,  7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   37,  87,  67,  23,  91,  7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   37,  87,  67,  23,  91,  7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   37,
      37,  67,  67,  39,  79,  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   37,  7,   67,  87,  23,  91,  7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      87,  87,  7,   103, 7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   71,  87,  45,  41,  77,  7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   87,
      87,  7,   103, 7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   37,  7,   67,  87,  23,  91,  7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   37,  7,   67,  87,
      23,  91,  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   71,  7,   45,  87,  41,  77,  7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   255, 135, 135, 255, 255, 143,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 7,   71,  7,   45,  87,  41,  77,  7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   37,  7,   67,  87,  23,  91,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   37,  7,   67,  87,  23,  91,  7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   87,  87,  7,   103, 7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   71,  87,  45,  41,  77,  7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   87,  87,  7,   103, 7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   37,
      7,   67,  87,  23,  91,  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   37,  37,  67,  67,  39,  79,  7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   37,
      87,  67,  23,  91,  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   37,  87,  67,  23,  91,  7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   37,  87,
      67,  23,  91,  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
      7,   7,   7,   7,   37,  37,  67,  67,  39,  79,  7,   7,   7,   7,   7,
      7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   99,  99,  99,  99,  99,
      99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,
      99,  99,  111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
      111, 111, 111, 111, 111, 111, 111, 111, 111,
  };
  const int ldc = m;
  int c_offset[] = {
      6477, 12954, 12954, 7793, 7793, 12954, 9282, 6477, 6477, 6477, 6477,
      6477, 6477,  6477,  6477, 6477, 6477,  6477, 6477, 6477, 6477, 6477,
  };
  int c_mult_int[] = {
      41121, 20560, 20560, 34267, 34267, 21937, 28784, 41121,
      41121, 41121, 41121, 41121, 41121, 41121, 41121, 41121,
      41121, 41121, 41121, 41121, 41121, 41121,
  };
  const int c_shift = 21;

  const int c_count = m * n;
  std::unique_ptr<std::uint8_t[]> output_data(new std::uint8_t[c_count]);
  MatrixMap<std::uint8_t, MapOrder::ColMajor> result(output_data.get(), m, n,
                                                     ldc);
  const OffsetColMap result_offset(c_offset, m);
  const OffsetColMap result_mult_int(c_mult_int, m);
  const int result_shift = c_shift;

  GemmContext gemm_context;
  auto output_pipeline = MakeStandardOutputPipeline<VectorShape::Col>(
      result_offset, result_mult_int, result_shift);
  GemmWithOutputPipelinePC<std::uint8_t, std::uint8_t,
                           DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, lhs_offset, rhs_offset,
      output_pipeline);

  ResultStats stats;
  GetResultStats(output_data.get(), expected_c_data, c_count, &stats);

  ResultStatsBounds bounds;
  const bool good = CheckResultStatsBounds(stats, bounds);
  printf("TestWithLargeDataPerChannelQuantization: %s\n",
         good ? "PASS" : "FAIL");
  ReportResultStats(stats, bounds);
  Check(good);
}

// Multithreading only activates when the result has more than 16 rows, and also
// (result rows) * (result cols) * depth >= 2 x 65 x 1024.  Size was selected
// to run in 3 threads.
//
// Based on the following floating point data:
//   LHS: all zeros except 10.0, 20.0 at the beginning of first 16 rows;
//     1.0, 2.0 at the beginning of next 16 rows; 0.1, 0.2 in next 16 rows;
//     0.01, 0.02 in last 16 rows.
//   RHS: all zeros except 1.0 in (0, 0) and 2.0 in (1, 0).
//   Varying boundaries were used for each 16 rows block of LHS, to test for
//     correct indexing into offsets.
//   Expected result: all zeros, except 50.0 at the beginning of first 16 rows;
//     5.0 at the beginning of next 16 rows; 0.5 in next 16 rows; 0.05 in last
//     16 rows.
void TestMultithreadedPerChannelQuantization() {
  const int m = 64;
  const int n = 20;
  const int k = 160;

  // LHS, m x k.
  const std::array<std::int32_t, 4> lhs_offsets_terse{{
      0, -51, -85, -109,
  }};
  assert(lhs_offsets_terse.size() * 16 == m);
  const std::array<std::uint8_t, 4> lhs_first_el{{
      128, 153, 170, 182,
  }};
  assert(lhs_first_el.size() * 16 == m);

  // lhs_first_el at (i, 0) and 255 at (i, 1), other values are all -offset.
  std::vector<std::uint8_t> a_data(m * k, 0);
  for (int i = 0; i < m; ++i) {
    a_data[i * k] = lhs_first_el[i / 16];
    a_data[i * k + 1] = 255;
    for (int j = 2; j < k; ++j) {
      a_data[i * k + j] = std::uint8_t(-lhs_offsets_terse[i / 16]);
    }
  }

  const int lda = k;
  // Given values at [i / 16].
  std::vector<std::int32_t> a_offset(m, 0);
  for (int i = 0; i < m; ++i) {
    a_offset[i] = lhs_offsets_terse[i / 16];
  }

  MatrixMap<const std::uint8_t, MapOrder::RowMajor> lhs(&a_data[0], m, k, lda);
  const OffsetColMap lhs_offset(&a_offset[0], m);

  // RHS, k x n.
  // All zeros, except 128 at (0, 0) and 255 at (1, 0).
  std::vector<std::uint8_t> b_data(k * n, 0);
  b_data[0] = 128;
  b_data[1] = 255;

  const int ldb = k;
  std::int32_t b_offset = 0;
  MatrixMap<const std::uint8_t, MapOrder::ColMajor> rhs(&b_data[0], k, n, ldb);
  const OffsetRowDup rhs_offset(b_offset, rhs.cols());

  // Result, m x n.
  // All zeros, except given values at (i / 16, 0).
  const std::array<std::uint8_t, 4> expected_c_terse{{
      142, 159, 182, 213,
  }};
  assert(expected_c_terse.size() * 16 == m);
  std::vector<std::uint8_t> expected_c_data(m * n, 0);
  for (int i = 0; i < m; ++i) {
    expected_c_data[i] = expected_c_terse[i / 16];
  }

  const int ldc = m;
  // All zeros.
  std::vector<std::int32_t> c_offset(m, 0);
  // Given values at [i / 16].
  const std::array<std::int32_t, 4> c_mult_int_terse{{
      3655, 5140, 7049, 9595,
  }};
  assert(c_mult_int_terse.size() * 16 == m);
  std::vector<std::int32_t> c_mult_int(m);
  for (int i = 0; i < m; ++i) {
    c_mult_int[i] = c_mult_int_terse[i / 16];
  }

  const int c_shift = 21;

  const int c_count = m * n;
  std::unique_ptr<std::uint8_t[]> output_data(new std::uint8_t[c_count]);
  MatrixMap<std::uint8_t, MapOrder::ColMajor> result(output_data.get(), m, n,
                                                     ldc);
  const OffsetColMap result_offset(&c_offset[0], m);
  const OffsetColMap result_mult_int(&c_mult_int[0], m);
  const int result_shift = c_shift;

  GemmContext gemm_context;
  auto output_pipeline = MakeStandardOutputPipeline<VectorShape::Col>(
      result_offset, result_mult_int, result_shift);
  GemmWithOutputPipelinePC<std::uint8_t, std::uint8_t,
                           DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, lhs_offset, rhs_offset,
      output_pipeline);

  ResultStats stats;
  GetResultStats(output_data.get(), &expected_c_data[0], c_count, &stats);

  ResultStatsBounds bounds;
  const bool good = CheckResultStatsBounds(stats, bounds);
  printf("TestMultithreadedPerChannelQuantization: %s\n",
         good ? "PASS" : "FAIL");
  ReportResultStats(stats, bounds);
  Check(good);
}

// Runs a small set of hand-calculated data through the implementation.
void TestWithSmallData() {
  const int m = 4;
  const int n = 2;
  const int k = 3;
  // Matrix A (LHS) is:
  // |  7 | 10 | 13 | 16 |
  // |  8 | 11 | 14 | 17 |
  // |  9 | 12 | 15 | 18 |
  const std::uint8_t a_data[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  // Matrix B (RHS) is:
  // |  1 |  3 |  5 |
  // |  2 |  4 |  6 |
  const std::uint8_t b_data[] = {1, 2, 3, 4, 5, 6};
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (3 * 8) + (5 * 9) = 76
  // (2 * 7) + (4 * 8) + (6 * 9) = 100
  // (1 * 10) + (3 * 11) + (5 * 12) = 103
  // (2 * 10) + (4 * 11) + (6 * 12) = 136
  // (1 * 13) + (3 * 14) + (5 * 15) = 130
  // (2 * 13) + (4 * 14) + (6 * 15) = 172
  // (1 * 16) + (3 * 17) + (5 * 18) = 157
  // (2 * 16) + (4 * 17) + (6 * 18) = 208
  // That means matrix C should be:
  // |  76 | 103 | 130 | 157 |
  // | 100 | 136 | 172 | 208 |
  const std::uint8_t expected_data[] = {76, 100, 103, 136, 130, 172, 157, 208};

  const int c_count = m * n;
  std::unique_ptr<std::uint8_t[]> output_data(new std::uint8_t[c_count]);

  const bool is_a_transposed = true;
  const bool is_b_transposed = true;
  const bool is_c_transposed = true;
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const int a_offset = 0;
  const int b_offset = 0;
  const int c_offset = 0;
  const int c_mult = 1;
  const int c_shift = 0;

  gemmlowp::eight_bit_int_gemm::EightBitIntGemm(
      is_a_transposed, is_b_transposed, is_c_transposed, m, n, k, a_data,
      a_offset, lda, b_data, b_offset, ldb, output_data.get(), c_offset, c_mult,
      c_shift, ldc, eight_bit_int_gemm::BitDepthSetting::A8B8);

  ResultStats stats;
  GetResultStats(output_data.get(), expected_data, c_count, &stats);

  ResultStatsBounds bounds;
  const bool good = CheckResultStatsBounds(stats, bounds);
  printf("TestWithSmallData: %s\n", good ? "PASS" : "FAIL");
  ReportResultStats(stats, bounds);
  Check(good);
}

// This is the most realistic test of how we'll be using the low-precision GEMM
// function in applications. It takes in large input matrices that have been
// captured from an actual neural network run.
void TestWithRealData(eight_bit_int_gemm::BitDepthSetting BitDepth,
                      int tolerance_median, int tolerance_max) {
  std::unique_ptr<std::uint8_t[]> output_data(
      new std::uint8_t[test_data::c_count]);
  gemmlowp::eight_bit_int_gemm::EightBitIntGemm(
      test_data::is_a_transposed, test_data::is_b_transposed,
      test_data::is_c_transposed, test_data::m, test_data::n, test_data::k,
      test_data::a_data, test_data::a_offset, test_data::k, test_data::b_data,
      test_data::b_offset, test_data::k, output_data.get(), test_data::c_offset,
      test_data::c_mult_int, test_data::c_shift, test_data::m, BitDepth);

  ResultStats stats;
  GetResultStats(output_data.get(), test_data::expected_c_data,
                 test_data::c_count, &stats);

  ResultStatsBounds bounds;
  if (BitDepth == eight_bit_int_gemm::BitDepthSetting::A5B7) {
    bounds.med_unsigned_diff = tolerance_median;
    bounds.max_unsigned_diff = tolerance_max;
    bounds.med_signed_diff = 0;
    bounds.mean_signed_diff = 0.2f;
  }

  const bool good = CheckResultStatsBounds(stats, bounds);
  printf("TestWithRealData: %s with %s\n", good ? "PASS" : "FAIL",
         GetBitDepthName(BitDepth));
  ReportResultStats(stats, bounds);
  Check(good);
}

template <typename BitDepthParams, MapOrder ResultOrder>
void TestOutputStages(int rows, int depth, int cols, int result_offset,
                      int result_mult_int, int result_shift) {
  Matrix<std::uint8_t, MapOrder::RowMajor> lhs(rows, depth);
  Matrix<std::uint8_t, MapOrder::ColMajor> rhs(depth, cols);
  Matrix<std::int32_t, ResultOrder> result_raw_int32(rows, cols);
  MakeRandom<typename BitDepthParams::LhsRange>(&lhs);
  MakeRandom<typename BitDepthParams::RhsRange>(&rhs);
  const int lhs_offset = 12;
  const int rhs_offset = -34;

  // Test an empty pipeline, i.e. returning raw int32 accumulators.
  auto empty_pipeline = std::make_tuple();
  GemmContext context;
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_raw_int32, lhs_offset,
      rhs_offset, empty_pipeline);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t expected = 0;
      for (int d = 0; d < depth; d++) {
        std::int32_t lhs_val =
            static_cast<std::int32_t>(lhs(r, d)) + lhs_offset;
        std::int32_t rhs_val =
            static_cast<std::int32_t>(rhs(d, c)) + rhs_offset;
        expected += lhs_val * rhs_val;
      }
      Check(expected == result_raw_int32(r, c));
    }
  }

  // Test a pipeline with only the quantize-down stage, still returning
  // unclamped (but scaled) int32's
  OutputStageQuantizeDownInt32ToUint8Scale quantize_down_stage;
  quantize_down_stage.result_offset = result_offset;
  quantize_down_stage.result_mult_int = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  auto quantize_down_pipeline = std::make_tuple(quantize_down_stage);
  Matrix<std::int32_t, ResultOrder> result_quantized_down_int32(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_quantized_down_int32,
      lhs_offset, rhs_offset, quantize_down_pipeline);

  std::int64_t sum = 0;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t raw = result_raw_int32(r, c);
      std::int32_t expected = RoundingDivideByPOT(
          (raw + result_offset) * result_mult_int, result_shift);
      Check(expected == result_quantized_down_int32(r, c));
      sum += expected;
    }
  }
  std::int64_t avg = sum / (rows * cols);
  // Test that the average quantized-down value falls reasonably in the
  // middle of the [0..255] range. Otherwise, the multiplier / shift need to be
  // adjusted.
  Check(avg >= 64 && avg <= 192);

  // Test the familiar default pipeline consisting of quantize-down and
  // clamp-and-cast-to-uint8.
  OutputStageSaturatingCastToUint8 saturating_cast_stage;
  auto quantize_down_and_saturating_cast_pipeline =
      std::make_tuple(quantize_down_stage, saturating_cast_stage);
  Matrix<std::uint8_t, ResultOrder> result_quantized_down_saturated_uint8(rows,
                                                                          cols);
  GemmWithOutputPipeline<std::uint8_t, std::uint8_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(),
      &result_quantized_down_saturated_uint8, lhs_offset, rhs_offset,
      quantize_down_and_saturating_cast_pipeline);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t quantized = result_quantized_down_int32(r, c);
      std::uint8_t expected = std::min(std::max(quantized, 0), 255);
      Check(expected == result_quantized_down_saturated_uint8(r, c));
    }
  }

  // Test a variant of the familiar default pipeline consisting of quantize-down
  // and clamp-and-cast-to-int16.
  OutputStageSaturatingCastToInt16 saturating_cast_int16_stage;
  auto quantize_down_and_saturating_cast_int16_pipeline =
      std::make_tuple(quantize_down_stage, saturating_cast_int16_stage);
  Matrix<std::int16_t, ResultOrder> result_quantized_down_saturated_int16(rows,
                                                                          cols);
  GemmWithOutputPipeline<std::uint8_t, std::int16_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(),
      &result_quantized_down_saturated_int16, lhs_offset, rhs_offset,
      quantize_down_and_saturating_cast_int16_pipeline);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t quantized = result_quantized_down_int32(r, c);
      std::int16_t expected = std::min(std::max(quantized, -32768), 32767);
      Check(expected == result_quantized_down_saturated_int16(r, c));
    }
  }

  // Test a bias-addition with row-vector
  std::vector<std::int32_t> row_vector_data(cols);
  std::uniform_int_distribution<std::int32_t> uniform_minus_500_plus_500(-500,
                                                                         500);
  for (int i = 0; i < cols; i++) {
    row_vector_data[i] = uniform_minus_500_plus_500(RandomEngine());
  }
  typedef VectorMap<std::int32_t, VectorShape::Row> RowVectorMap;
  RowVectorMap row_vector_map(row_vector_data.data(), cols);
  OutputStageBiasAddition<RowVectorMap> row_bias_addition_stage;
  row_bias_addition_stage.bias_vector = row_vector_map;
  auto row_bias_addition_pipeline = std::make_tuple(row_bias_addition_stage);
  Matrix<std::int32_t, ResultOrder> result_of_row_bias_addition(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_of_row_bias_addition,
      lhs_offset, rhs_offset, row_bias_addition_pipeline);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t expected = result_raw_int32(r, c) + row_vector_data[c];
      Check(expected == result_of_row_bias_addition(r, c));
    }
  }

  // Test a bias-addition with column-vector
  std::vector<std::int32_t> col_vector_data(rows);
  for (int i = 0; i < rows; i++) {
    col_vector_data[i] = uniform_minus_500_plus_500(RandomEngine());
  }
  typedef VectorMap<std::int32_t, VectorShape::Col> ColVectorMap;
  ColVectorMap col_vector_map(col_vector_data.data(), rows);
  OutputStageBiasAddition<ColVectorMap> col_bias_addition_stage;
  col_bias_addition_stage.bias_vector = col_vector_map;
  auto col_bias_addition_pipeline = std::make_tuple(col_bias_addition_stage);
  Matrix<std::int32_t, ResultOrder> result_of_col_bias_addition(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_of_col_bias_addition,
      lhs_offset, rhs_offset, col_bias_addition_pipeline);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t expected = result_raw_int32(r, c) + col_vector_data[r];
      Check(expected == result_of_col_bias_addition(r, c));
    }
  }

  // Test a clamp
  OutputStageClamp clamp_stage;
  // Determine min and max of raw int32 accumulators
  std::int32_t raw_min = std::numeric_limits<std::int32_t>::max();
  std::int32_t raw_max = std::numeric_limits<std::int32_t>::min();
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      raw_min = std::min(raw_min, result_raw_int32(r, c));
      raw_max = std::max(raw_max, result_raw_int32(r, c));
    }
  }
  // Pick some interesting clamp min/max bounds
  clamp_stage.min = static_cast<std::int32_t>(raw_min * 0.7 + raw_max * 0.3);
  clamp_stage.max = static_cast<std::int32_t>(raw_min * 0.3 + raw_max * 0.7);
  assert(raw_min <= clamp_stage.min && clamp_stage.min <= clamp_stage.max &&
         clamp_stage.max <= raw_max);
  auto clamp_pipeline = std::make_tuple(clamp_stage);
  Matrix<std::int32_t, ResultOrder> result_clamped(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_clamped, lhs_offset,
      rhs_offset, clamp_pipeline);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t raw = result_raw_int32(r, c);
      std::int32_t expected =
          std::min(std::max(raw, clamp_stage.min), clamp_stage.max);
      Check(expected == result_clamped(r, c));
    }
  }

  // Test tanh
  OutputStageTanh tanh_stage;
  const std::int32_t real_zero_as_int32 = (raw_max + raw_min) / 2;
  const std::int32_t real_amplitude_as_int32 = (raw_max - raw_min) / 16;
  tanh_stage.real_zero_as_int32 = real_zero_as_int32;
  tanh_stage.real_amplitude_as_int32 = real_amplitude_as_int32;
  auto tanh_pipeline = std::make_tuple(tanh_stage);
  Matrix<std::int32_t, ResultOrder> result_tanh(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_tanh, lhs_offset,
      rhs_offset, tanh_pipeline);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t raw = result_raw_int32(r, c);
      double real_input =
          double(raw - real_zero_as_int32) / real_amplitude_as_int32;
      double expected = std::tanh(real_input);
      std::int32_t actual_int32 = result_tanh(r, c);
      double actual =
          double(actual_int32 - real_zero_as_int32) / real_amplitude_as_int32;
      Check(std::abs(expected - actual) < 2e-4);
    }
  }

  // Test a pipeline with bias and clamp
  auto bias_clamp_pipeline =
      std::make_tuple(col_bias_addition_stage, clamp_stage);
  Matrix<std::int32_t, ResultOrder> result_biased_clamped(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(), &result_biased_clamped,
      lhs_offset, rhs_offset, bias_clamp_pipeline);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t raw = result_raw_int32(r, c);
      std::int32_t biased = raw + col_vector_data[r];
      std::int32_t expected =
          std::min(std::max(biased, clamp_stage.min), clamp_stage.max);
      Check(expected == result_biased_clamped(r, c));
    }
  }

  // Test a full pipeline with bias and clamp and quantization down to 8bit
  // result
  auto bias_clamp_quantize_cast_pipeline =
      std::make_tuple(col_bias_addition_stage, clamp_stage, quantize_down_stage,
                      saturating_cast_stage);
  Matrix<std::uint8_t, ResultOrder> result_biased_clamped_quantized_casted(
      rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::uint8_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(),
      &result_biased_clamped_quantized_casted, lhs_offset, rhs_offset,
      bias_clamp_quantize_cast_pipeline);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t quantized = RoundingDivideByPOT(
          (result_biased_clamped(r, c) + result_offset) * result_mult_int,
          result_shift);
      std::uint8_t expected = std::min(std::max(quantized, 0), 255);
      Check(expected == result_biased_clamped_quantized_casted(r, c));
    }
  }

  // Test a pipeline with the fixed-point-multiplier variant stage for the
  // quantizing down of 32bit accumulators.
  //
  // First, figure appropriate fixedpoint multiplier and shift values.
  std::int32_t result_fixedpoint_multiplier = result_mult_int;
  std::int32_t result_fixedpoint_shift = result_shift;
  Check(result_mult_int > 0);
  Check(result_shift > 0);
  result_fixedpoint_multiplier = result_mult_int;
  result_fixedpoint_shift = result_shift - 31;
  while (result_fixedpoint_multiplier < (1 << 30)) {
    result_fixedpoint_multiplier <<= 1;
    result_fixedpoint_shift++;
  }
  Check(result_fixedpoint_shift >= 0);
  // Now test OutputStageQuantizeDownInt32ByFixedPoint
  OutputStageQuantizeDownInt32ByFixedPoint
      quantize_down_by_fixedpoint_stage;
  quantize_down_by_fixedpoint_stage.result_offset_after_shift =
      static_cast<std::int32_t>(
          round(static_cast<double>(result_offset * result_mult_int) /
                (1 << result_shift)));
  quantize_down_by_fixedpoint_stage.result_fixedpoint_multiplier =
      result_fixedpoint_multiplier;
  quantize_down_by_fixedpoint_stage.result_shift = result_fixedpoint_shift;
  auto quantize_down_by_fixedpoint_pipeline =
      std::make_tuple(quantize_down_by_fixedpoint_stage);
  Matrix<std::int32_t, ResultOrder> result_quantized_down_by_fixedpoint_int32(
      rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::int32_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(),
      &result_quantized_down_by_fixedpoint_int32, lhs_offset, rhs_offset,
      quantize_down_by_fixedpoint_pipeline);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const std::int32_t actual =
          result_quantized_down_by_fixedpoint_int32(r, c);
      const std::int32_t raw = result_raw_int32(r, c);
      const std::int32_t expected =
          quantize_down_by_fixedpoint_stage.result_offset_after_shift +
          RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                  raw, result_fixedpoint_multiplier),
                              result_fixedpoint_shift);
      Check(actual == expected);
    }
  }

  // Test OutputStageScaleInt32ByFixedPointAndExponent
  for (int exponent = -2; exponent <= 2; exponent++) {
    OutputStageScaleInt32ByFixedPointAndExponent
        scale_by_fixedpoint_and_exponent_stage;
    scale_by_fixedpoint_and_exponent_stage.result_offset_after_shift =
        static_cast<std::int32_t>(round(static_cast<double>(
            result_offset * result_mult_int * std::pow(2.0, exponent))));
    scale_by_fixedpoint_and_exponent_stage.result_fixedpoint_multiplier =
        result_fixedpoint_multiplier;
    scale_by_fixedpoint_and_exponent_stage.result_exponent = exponent;
    auto scale_by_fixedpoint_and_exponent_pipeline =
        std::make_tuple(scale_by_fixedpoint_and_exponent_stage);
    Matrix<std::int32_t, ResultOrder>
        result_scaled_by_fixedpoint_and_exponent_int32(rows, cols);
    GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                           DefaultL8R8BitDepthParams>(
        &context, lhs.const_map(), rhs.const_map(),
        &result_scaled_by_fixedpoint_and_exponent_int32, lhs_offset, rhs_offset,
        scale_by_fixedpoint_and_exponent_pipeline);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        const std::int32_t actual =
            result_scaled_by_fixedpoint_and_exponent_int32(r, c);
        const std::int32_t raw = result_raw_int32(r, c);
        int left_shift = std::max(0, exponent);
        int right_shift = std::max(0, -exponent);
        const std::int32_t expected =
            scale_by_fixedpoint_and_exponent_stage.result_offset_after_shift +
            RoundingDivideByPOT(
                SaturatingRoundingDoublingHighMul((1 << left_shift) * raw,
                                                  result_fixedpoint_multiplier),
                right_shift);
        Check(actual == expected);
      }
    }
  }

  // Test the variant of the familiar default pipeline consisting of
  // quantize-down and
  // clamp-and-cast-to-uint8, where we used fixedpoint multipliers for the
  // downscaling.
  auto quantize_down_by_fixedpoint_and_saturating_cast_pipeline =
      std::make_tuple(quantize_down_by_fixedpoint_stage, saturating_cast_stage);
  Matrix<std::uint8_t, ResultOrder>
      result_quantized_down_by_fixedpoint_saturated_uint8(rows, cols);
  GemmWithOutputPipeline<std::uint8_t, std::uint8_t, DefaultL8R8BitDepthParams>(
      &context, lhs.const_map(), rhs.const_map(),
      &result_quantized_down_by_fixedpoint_saturated_uint8, lhs_offset,
      rhs_offset, quantize_down_by_fixedpoint_and_saturating_cast_pipeline);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::int32_t quantized = result_quantized_down_by_fixedpoint_int32(r, c);
      std::uint8_t expected = std::min(std::max(quantized, 0), 255);
      Check(expected ==
            result_quantized_down_by_fixedpoint_saturated_uint8(r, c));
    }
  }

  printf("TestOutputStages: PASS with ResultOrder=%s\n",
         OrderName(ResultOrder));
}

#ifndef GEMMLOWP_SKIP_EXHAUSTIVE_TESTS
template <typename BitDepthParams>
void TestExhaustively() {
  GemmContext context;

  // Test the internal GEMM interfaces
  test_gemm<
      SingleThreadGemmWrapper<DefaultKernel<BitDepthParams>,
                              std::uint8_t, BitDepthParams>>(&context);

  test_gemm<
      MultiThreadGemmWrapper<DefaultKernel<BitDepthParams>,
                             std::uint8_t, BitDepthParams>>(&context);

  // Test the public GEMM interfaces
  test_gemm<PublicGemmWrapper<std::uint8_t, BitDepthParams>>(&context);

  // Test GEMV cases (internal interfaces)
  test_gemv<
      SingleThreadGemmWrapper<DefaultKernel<BitDepthParams>,
                              std::uint8_t, BitDepthParams>>(&context);

  test_gemv<
      MultiThreadGemmWrapper<DefaultKernel<BitDepthParams>,
                             std::uint8_t, BitDepthParams>>(&context);

  // Test GEMV cases (public interfaces)
  test_gemv<PublicGemmWrapper<std::uint8_t, BitDepthParams>>(&context);
}

template <eight_bit_int_gemm::BitDepthSetting BitDepthSetting>
void TestExhaustivelyEightBitIntGemm() {
  GemmContext context;
  test_gemv<EightBitIntGemmWrapper<std::uint8_t, BitDepthSetting>>(&context);
  test_gemv<EightBitIntGemmWrapper<std::uint8_t, BitDepthSetting>>(&context);
  test_gemm<EightBitIntGemmWrapper<std::uint8_t, BitDepthSetting>>(&context);
}

void TestKernels() {
  GemmContext context;

  // Test specific kernels with various different formats,
  // to exercises corner cases especially in the packing code.
  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<1, 1>, 1>,
                                   KernelSideFormat<CellFormat<1, 1>, 1>>>>(
      &context);

  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 2>, 1>,
                                   KernelSideFormat<CellFormat<4, 2>, 2>>>>(
      &context);

  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 2>, 4>,
                                   KernelSideFormat<CellFormat<4, 2>, 5>>>>(
      &context);

  test_gemm_kernel<ReferenceKernel<KernelFormat<
      KernelSideFormat<CellFormat<3, 4, CellOrder::DepthMajor>, 2>,
      KernelSideFormat<CellFormat<5, 4, CellOrder::DepthMajor>, 3>>>>(&context);

  test_gemm_kernel<ReferenceKernel<KernelFormat<
      KernelSideFormat<CellFormat<3, 4, CellOrder::WidthMajor>, 2>,
      KernelSideFormat<CellFormat<5, 4, CellOrder::WidthMajor>, 3>>>>(&context);

  test_gemm_kernel<ReferenceKernel<KernelFormat<
      KernelSideFormat<CellFormat<5, 2, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 2>>>>(&context);

  test_gemm_kernel<ReferenceKernel<KernelFormat<
      KernelSideFormat<CellFormat<5, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 2>>>>(&context);

  test_gemm_kernel<ReferenceKernel<KernelFormat<
      KernelSideFormat<CellFormat<8, 8, CellOrder::Diagonal>, 2>,
      KernelSideFormat<CellFormat<3, 8, CellOrder::WidthMajor>, 1>>>>(&context);

  test_gemm_kernel<ReferenceKernel<KernelFormat<
      KernelSideFormat<CellFormat<1, 4, CellOrder::DepthMajor>, 1>,
      KernelSideFormat<CellFormat<4, 4, CellOrder::Diagonal>, 1>>>>(&context);
}

#endif  // not GEMMLOWP_SKIP_EXHAUSTIVE_TESTS

template <typename BitDepthParams>
void TestOutputStages() {
  // Test non-default output pipelines with various combinations of
  // output stages.
  TestOutputStages<BitDepthParams, MapOrder::RowMajor>(63, 10, 127, 5, 17, 14);
  TestOutputStages<BitDepthParams, MapOrder::ColMajor>(63, 10, 127, 5, 17, 14);
  TestOutputStages<BitDepthParams, MapOrder::RowMajor>(630, 10, 1270, 5, 17,
                                                       14);
  TestOutputStages<BitDepthParams, MapOrder::ColMajor>(630, 10, 1270, 5, 17,
                                                       14);
}

void test() {
#ifdef GEMMLOWP_TEST_PROFILE
  RegisterCurrentThreadForProfiling();
  StartProfiling();
#endif

  // Run a first quick test against hand-calculated data.
  TestWithSmallData();

#ifndef GEMMLOWP_SKIP_EXHAUSTIVE_TESTS
  TestExhaustively<DefaultL8R8BitDepthParams>();
  TestExhaustively<L8R8WithLhsNonzeroBitDepthParams>();
  TestExhaustively<DefaultL7R5BitDepthParams>();  // legacy, same as L8R8
  TestExhaustivelyEightBitIntGemm<eight_bit_int_gemm::BitDepthSetting::A8B8>();
  TestExhaustivelyEightBitIntGemm<eight_bit_int_gemm::BitDepthSetting::A5B7>();
  TestKernels();
#endif

  // Run against actual data from a network evaluation.
  TestWithRealData(eight_bit_int_gemm::BitDepthSetting::A8B8, 0, 0);
  TestWithRealData(eight_bit_int_gemm::BitDepthSetting::A5B7, 2, 10);

  // Test non-default output pipelines with various combinations of
  // output stages.
  TestOutputStages<DefaultL8R8BitDepthParams>();
  TestOutputStages<L8R8WithLhsNonzeroBitDepthParams>();

  // Test per channel quantization.
  TestWithSmallDataPerChannelQuantization();
  TestWithLargeDataPerChannelQuantization();
  TestMultithreadedPerChannelQuantization();
#ifdef GEMMLOWP_TEST_PROFILE
  FinishProfiling();
#endif

  std::cerr << "All tests passed." << std::endl;

  // We have been testing the eight_bit_int_gemm, so we should free its
  // persistent
  // resources now to avoid having leak-checking tools report leaks.
  eight_bit_int_gemm::FreePersistentResources();
}

}  // end namespace gemmlowp

// For iOS, we need to define our own main(), so skip it here.
#if !(defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR))
int main() { gemmlowp::test(); }
#endif
