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

// test_fixedpoint.cc: unit tests covering the fixedpoint/ directory.

#define GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cinttypes>
#include <random>
#include <vector>
#include "test.h"

#include "../fixedpoint/fixedpoint.h"

namespace gemmlowp {

namespace {

template <typename T>
T Load(const typename FixedPointRawTypeTraits<T>::ScalarRawType* src) {
  return *src;
}
template <typename T>
void Store(typename FixedPointRawTypeTraits<T>::ScalarRawType* dst, T v) {
  *dst = v;
}
#ifdef GEMMLOWP_NEON
template <>
int32x4_t Load<int32x4_t>(const std::int32_t* src) {
  return vld1q_s32(src);
}
template <>
int16x8_t Load<int16x8_t>(const std::int16_t* src) {
  return vld1q_s16(src);
}
template <>
void Store<int32x4_t>(std::int32_t* dst, int32x4_t v) {
  vst1q_s32(dst, v);
}
template <>
void Store<int16x8_t>(std::int16_t* dst, int16x8_t v) {
  vst1q_s16(dst, v);
}
#endif
#ifdef GEMMLOWP_SSE4
template <>
__m128i Load<__m128i>(const std::int32_t* src) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
}
template <>
void Store<__m128i>(std::int32_t* dst, __m128i v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v);
}
template <>
int16x8_m128i Load<int16x8_m128i>(const std::int16_t* src) {
  return int16x8_m128i(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
}
template <>
void Store<int16x8_m128i>(std::int16_t* dst, int16x8_m128i v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v.v);
}
#endif
#ifdef GEMMLOWP_MSA
template <>
v4i32 Load<v4i32>(const std::int32_t* src) {
  return __builtin_msa_ld_w(const_cast<std::int32_t*>(src), 0);
}
template <>
v8i16 Load<v8i16>(const std::int16_t* src) {
  return __builtin_msa_ld_h(const_cast<std::int16_t*>(src), 0);
}
template <>
void Store<v4i32>(std::int32_t* dst, v4i32 v) {
  __builtin_msa_st_w(v, dst, 0);
}
template <>
void Store<v8i16>(std::int16_t* dst, v8i16 v) {
  __builtin_msa_st_h(v, dst, 0);
}
#endif

template <typename tSimdType>
class TestFixedPoint {
 public:
  using SimdType = tSimdType;
  using SimdTypeTraits = FixedPointRawTypeTraits<SimdType>;
  using ScalarType = typename SimdTypeTraits::ScalarRawType;
  static constexpr int kSimdLanes = SimdTypeTraits::kLanes;
  static constexpr int kScalarTypeBits = 8 * sizeof(ScalarType);

  // Explanation of UnaryOpBase, its *Op subclasses below, and TestUnaryOp:
  // Most (though not all) of the fixedpoint functionality being tested
  // consists of functions taking one fixedpoint value and returning one
  // fixedpoint value, e.g. "exp" or "tanh". We call them "unary operators".
  // We factor a lot of testing boilerplate into a common TestUnaryOp function
  // taking a "unary op" object that fully describes the function to be tested.
  // These objects inherit UnaryOpBase mostly as a means to share some default
  // values for some properties.
  //
  // An important design element here is that the fixed-point values are passed
  // around as raw integers (e.g. int32_t or SIMD types such as int32x4_t), not
  // as higher-level FixedPoint objects. The motivation for this design is 1) to
  // avoid having to templatize everything in the tIntegerBits parameter of
  // class FixedPoint, and 2) to allow directly testing low-level functions
  // operating on raw types (e.g. RoundingDivideByPOT) without needlessly
  // requiring
  // wrapping raw values in FixedPoint objects.
  class UnaryOpBase {
   public:
    // Min bound of the input range of this op. For example, an op only handling
    // nonnegative values would return 0.
    ScalarType MinInput() const {
      return std::numeric_limits<ScalarType>::min();
    }
    // Max bound of the input range of this op. For example, an op only handling
    // nonpositive values would return 0.
    ScalarType MaxInput() const {
      return std::numeric_limits<ScalarType>::max();
    }
    // Tolerated difference between actual and reference ScalarType values.
    // Note that the corresponding real-numbers tolerance depends on the number
    // of integer bits of the fixed-point representation of the results of this
    // op.
    // For example, for an op returning fixed-point values with 0 integer bits,
    // the correspondence between real-number values and raw values is
    // real_number = (2^31) * raw_value.
    ScalarType Tolerance() const { return 0; }
  };

  // Op wrapping RoundingDivideByPOT
  class RoundingDivideByPOTOp final : public UnaryOpBase {
   public:
    RoundingDivideByPOTOp(int exponent) : exponent_(exponent) {}
    ScalarType ReferenceOp(ScalarType x) const {
      const double d = static_cast<double>(x) / (1ll << exponent_);
      return static_cast<ScalarType>(std::round(d));
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      return RoundingDivideByPOT(x, exponent_);
    }

   private:
    const int exponent_;
  };

  // Op wrapping SaturatingRoundingMultiplyByPOT
  template <int tExponent>
  class SaturatingRoundingMultiplyByPOTOp final : public UnaryOpBase {
   public:
    ScalarType ReferenceOp(ScalarType x) const {
      const double d = static_cast<double>(x) * std::pow(2., tExponent);
      const double clamp_min = std::numeric_limits<ScalarType>::min();
      const double clamp_max = std::numeric_limits<ScalarType>::max();
      const double clamped = std::min(clamp_max, std::max(clamp_min, d));
      return static_cast<ScalarType>(std::round(clamped));
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      return SaturatingRoundingMultiplyByPOT<tExponent>(x);
    }
  };

  // Op wrapping exp_on_interval_between_negative_one_quarter_and_0_excl
  class ExpOnIntervalBetweenNegativeOneQuarterAnd0ExclOp final
      : public UnaryOpBase {
   public:
    ScalarType MinInput() const { return -(1 << (kScalarTypeBits - 3)); }
    ScalarType MaxInput() const { return 0; }
    ScalarType Tolerance() const { return kScalarTypeBits == 32 ? 500 : 1; }
    ScalarType ReferenceOp(ScalarType x) const {
      using F = FixedPoint<ScalarType, 0>;
      const double d = ToDouble(F::FromRaw(x));
      const double e = std::exp(d);
      return F::FromDouble(e).raw();
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      using F = FixedPoint<RawType, 0>;
      const F f = F::FromRaw(x);
      const F e = exp_on_interval_between_negative_one_quarter_and_0_excl(f);
      return e.raw();
    }
  };

  // Op wrapping exp_on_negative_values
  template <int tIntegerBits>
  class ExpOnNegativeValuesOp final : public UnaryOpBase {
   public:
    ScalarType MaxInput() const { return 0; }
    ScalarType Tolerance() const { return kScalarTypeBits == 32 ? 500 : 2; }
    ScalarType ReferenceOp(ScalarType x) const {
      using F = FixedPoint<ScalarType, tIntegerBits>;
      using F0 = FixedPoint<ScalarType, 0>;
      const double d = ToDouble(F::FromRaw(x));
      const double e = std::exp(d);
      return F0::FromDouble(e).raw();
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      using F = FixedPoint<RawType, tIntegerBits>;
      const F f = F::FromRaw(x);
      return exp_on_negative_values(f).raw();
    }
  };

  // Op wrapping one_minus_x_over_one_plus_x_for_x_in_0_1
  class OneMinusXOverOnePlusXForXIn01Op final : public UnaryOpBase {
   public:
    ScalarType MinInput() const { return 0; }
    ScalarType Tolerance() const { return kScalarTypeBits == 32 ? 12 : 11; }
    ScalarType ReferenceOp(ScalarType x) const {
      using F = FixedPoint<ScalarType, 0>;
      const double d = ToDouble(F::FromRaw(x));
      const double e = (1 - d) / (1 + d);
      return F::FromDouble(e).raw();
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      using F = FixedPoint<RawType, 0>;
      const F f = F::FromRaw(x);
      return one_minus_x_over_one_plus_x_for_x_in_0_1(f).raw();
    }
  };

  // Op wrapping tanh
  template <int tIntegerBits>
  class TanhOp final : public UnaryOpBase {
   public:
    ScalarType Tolerance() const { return kScalarTypeBits == 32 ? 310 : 12; }
    ScalarType ReferenceOp(ScalarType x) const {
      using F = FixedPoint<ScalarType, tIntegerBits>;
      using F0 = FixedPoint<ScalarType, 0>;
      const double d = ToDouble(F::FromRaw(x));
      const double e = std::tanh(d);
      return F0::FromDouble(e).raw();
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      using F = FixedPoint<RawType, tIntegerBits>;
      const F f = F::FromRaw(x);
      return tanh(f).raw();
    }
  };

  // Op wrapping one_over_one_plus_x_for_x_in_0_1
  class OneOverOnePlusXForXIn01Op final : public UnaryOpBase {
   public:
    ScalarType MinInput() const { return 0; }
    ScalarType Tolerance() const { return kScalarTypeBits == 32 ? 6 : 5; }
    ScalarType ReferenceOp(ScalarType x) const {
      using F = FixedPoint<ScalarType, 0>;
      const double d = ToDouble(F::FromRaw(x));
      const double e = 1 / (1 + d);
      return F::FromDouble(e).raw();
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      using F = FixedPoint<RawType, 0>;
      const F f = F::FromRaw(x);
      return one_over_one_plus_x_for_x_in_0_1(f).raw();
    }
  };

  // Op wrapping logistic
  template <int tIntegerBits>
  class LogisticOp final : public UnaryOpBase {
   public:
    ScalarType Tolerance() const { return kScalarTypeBits == 32 ? 155 : 6; }
    ScalarType ReferenceOp(ScalarType x) const {
      using F = FixedPoint<ScalarType, tIntegerBits>;
      using F0 = FixedPoint<ScalarType, 0>;
      const double d = ToDouble(F::FromRaw(x));
      const double e = 1 / (1 + std::exp(-d));
      return F0::FromDouble(e).raw();
    }
    template <typename RawType>
    RawType Op(RawType x) const {
      using F = FixedPoint<RawType, tIntegerBits>;
      const F f = F::FromRaw(x);
      return logistic(f).raw();
    }
  };

  // Tests a given op, on a given list of int32 input values.
  template <typename tUnaryOpType>
  void TestUnaryOp(const tUnaryOpType& unary_op,
                   const std::vector<ScalarType>& testvals) {
    Check(0 == (testvals.size() % kSimdLanes));
    for (std::size_t i = 0; i < testvals.size(); i += kSimdLanes) {
      // First, clamp input values accoding to the MinInput() and MaxInput()
      // bounds returned by the op.
      ScalarType input[kSimdLanes] = {0};
      for (std::size_t j = 0; j < kSimdLanes; j++) {
        const ScalarType raw_input = testvals[i + j];
        input[j] = std::min(unary_op.MaxInput(),
                            std::max(unary_op.MinInput(), raw_input));
      }
      // Compute reference results and check that the actual results on
      // scalar inputs agree with them, to the Tolerance() returned by the op.
      ScalarType reference[kSimdLanes] = {0};
      ScalarType actual_scalar[kSimdLanes] = {0};
      for (std::size_t j = 0; j < kSimdLanes; j++) {
        reference[j] = unary_op.ReferenceOp(input[j]);
        actual_scalar[j] = unary_op.Op(input[j]);
        const std::int64_t diff = static_cast<std::int64_t>(actual_scalar[j]) -
                                  static_cast<std::int64_t>(reference[j]);
        if (std::abs(diff) > unary_op.Tolerance()) {
          fprintf(stderr, "abs(diff) (%" PRId64 ") > tolerance (%d)\n", diff,
                  unary_op.Tolerance());
        }
        Check(std::abs(diff) <= unary_op.Tolerance());
      }
      // Check that the actual results on SIMD inputs agree *exactly* with the
      // actual results on scalar inputs. I.e. SIMD must make absolutely no
      // difference
      // to the results, regardless of the fact that both scalar and SIMD
      // results may differ from the reference results.
      ScalarType actual_simd[kSimdLanes] = {0};
      Store<SimdType>(actual_simd, unary_op.Op(Load<SimdType>(input)));
      for (std::size_t j = 0; j < kSimdLanes; j++) {
        if (actual_simd[j] != actual_scalar[j]) {
          fprintf(stderr, "SIMD (%d) != scalar (%d)\n", actual_simd[j],
                  actual_scalar[j]);
        }
        Check(actual_simd[j] == actual_scalar[j]);
      }
    }
  }

  template <int tIntegerBits>
  void test_convert(FixedPoint<ScalarType, tIntegerBits> x) {
    typedef FixedPoint<ScalarType, tIntegerBits> F;
    F y = F::FromDouble(ToDouble(x));
    Check(y == x);
  }

  template <int tIntegerBits_a, int tIntegerBits_b>
  void test_Rescale(FixedPoint<ScalarType, tIntegerBits_a> a) {
    FixedPoint<ScalarType, tIntegerBits_b> actual = Rescale<tIntegerBits_b>(a);
    FixedPoint<ScalarType, tIntegerBits_b> expected =
        FixedPoint<ScalarType, tIntegerBits_b>::FromDouble(ToDouble(a));
    Check(actual == expected);
  }

  template <int tIntegerBits_a, int tIntegerBits_b>
  void test_Rescale(const std::vector<ScalarType>& testvals) {
    for (auto a : testvals) {
      FixedPoint<ScalarType, tIntegerBits_a> aq;
      aq.raw() = a;
      test_Rescale<tIntegerBits_a, tIntegerBits_b>(aq);
    }
  }

  template <int tIntegerBits_a, int tIntegerBits_b>
  void test_mul(FixedPoint<ScalarType, tIntegerBits_a> a,
                FixedPoint<ScalarType, tIntegerBits_b> b) {
    static const int ProductIntegerBits = tIntegerBits_a + tIntegerBits_b;
    using ProductFixedPoint = FixedPoint<ScalarType, ProductIntegerBits>;
    ProductFixedPoint ab;
    ab = a * b;
    double a_double = ToDouble(a);
    double b_double = ToDouble(b);
    double ab_double = a_double * b_double;
    ProductFixedPoint expected = ProductFixedPoint::FromDouble(ab_double);
    std::int64_t diff = std::int64_t(ab.raw()) - std::int64_t(expected.raw());
    Check(std::abs(diff) <= 1);
  }

  template <int tIntegerBits_a, int tIntegerBits_b>
  void test_mul(const std::vector<ScalarType>& testvals) {
    for (auto a : testvals) {
      for (auto b : testvals) {
        FixedPoint<ScalarType, tIntegerBits_a> aq;
        FixedPoint<ScalarType, tIntegerBits_b> bq;
        aq.raw() = a;
        bq.raw() = b;
        test_mul(aq, bq);
      }
    }
  }

  template <int tExponent, int tIntegerBits_a>
  void test_ExactMulByPot(FixedPoint<ScalarType, tIntegerBits_a> a) {
    double x = ToDouble(a) * std::pow(2.0, tExponent);
    double y = ToDouble(ExactMulByPot<tExponent>(a));
    Check(x == y);
  }

  template <int tExponent, int tIntegerBits_a>
  void test_ExactMulByPot(const std::vector<ScalarType>& testvals) {
    for (auto a : testvals) {
      FixedPoint<ScalarType, tIntegerBits_a> aq;
      aq.raw() = a;
      test_ExactMulByPot<tExponent, tIntegerBits_a>(aq);
    }
  }

  // Make the list of test values to test each op against.
  std::vector<ScalarType> MakeTestVals() {
    std::vector<ScalarType> testvals;

    for (int i = 0; i < kScalarTypeBits - 1; i++) {
      testvals.push_back((1 << i) - 2);
      testvals.push_back((1 << i) - 1);
      testvals.push_back((1 << i));
      testvals.push_back((1 << i) + 1);
      testvals.push_back((1 << i) + 2);
      testvals.push_back(-(1 << i) - 2);
      testvals.push_back(-(1 << i) - 1);
      testvals.push_back(-(1 << i));
      testvals.push_back(-(1 << i) + 1);
      testvals.push_back(-(1 << i) + 2);
    }
    testvals.push_back(std::numeric_limits<ScalarType>::min());
    testvals.push_back(std::numeric_limits<ScalarType>::min() + 1);
    testvals.push_back(std::numeric_limits<ScalarType>::min() + 2);
    testvals.push_back(std::numeric_limits<ScalarType>::max() - 2);
    testvals.push_back(std::numeric_limits<ScalarType>::max() - 1);
    testvals.push_back(std::numeric_limits<ScalarType>::max());

    std::mt19937 random_engine;
    std::uniform_int_distribution<ScalarType> uniform_distribution(
        std::numeric_limits<ScalarType>::min(),
        std::numeric_limits<ScalarType>::max());
    for (int i = 0; i < 1000; i++) {
      testvals.push_back(uniform_distribution(random_engine));
    }

    // SIMD tests will require the length of testvals to be a multiple
    // of SIMD vector size.
    while (testvals.size() % kSimdLanes) {
      testvals.push_back(0);
    }

    std::sort(testvals.begin(), testvals.end());
    return testvals;
  }

  void RunTests(const char* msg) {
    const std::vector<ScalarType> testvals = MakeTestVals();

    for (int s = 0; s < kScalarTypeBits; s++) {
      TestUnaryOp(RoundingDivideByPOTOp(s), testvals);
    }

    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<1 - kScalarTypeBits>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<2 - kScalarTypeBits>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<3 - kScalarTypeBits>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<14 - kScalarTypeBits>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<15 - kScalarTypeBits>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-15>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-4>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-3>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-2>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-1>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<0>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<1>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<2>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<3>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<4>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<15>(), testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<kScalarTypeBits - 15>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<kScalarTypeBits - 14>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<kScalarTypeBits - 3>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<kScalarTypeBits - 2>(),
                testvals);
    TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<kScalarTypeBits - 1>(),
                testvals);

    TestUnaryOp(ExpOnIntervalBetweenNegativeOneQuarterAnd0ExclOp(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<0>(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<1>(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<2>(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<3>(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<4>(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<5>(), testvals);
    TestUnaryOp(ExpOnNegativeValuesOp<6>(), testvals);

    TestUnaryOp(OneMinusXOverOnePlusXForXIn01Op(), testvals);
    TestUnaryOp(TanhOp<0>(), testvals);
    TestUnaryOp(TanhOp<1>(), testvals);
    TestUnaryOp(TanhOp<2>(), testvals);
    TestUnaryOp(TanhOp<3>(), testvals);
    TestUnaryOp(TanhOp<4>(), testvals);
    TestUnaryOp(TanhOp<5>(), testvals);
    TestUnaryOp(TanhOp<6>(), testvals);

    TestUnaryOp(OneOverOnePlusXForXIn01Op(), testvals);
    TestUnaryOp(LogisticOp<0>(), testvals);
    TestUnaryOp(LogisticOp<1>(), testvals);
    TestUnaryOp(LogisticOp<2>(), testvals);
    TestUnaryOp(LogisticOp<3>(), testvals);
    TestUnaryOp(LogisticOp<4>(), testvals);
    TestUnaryOp(LogisticOp<5>(), testvals);
    TestUnaryOp(LogisticOp<6>(), testvals);

    for (auto a : testvals) {
      FixedPoint<ScalarType, 4> x;
      x.raw() = a;
      test_convert(x);
    }

    test_mul<0, 0>(testvals);
    test_mul<0, 1>(testvals);
    test_mul<2, 0>(testvals);
    test_mul<1, 1>(testvals);
    test_mul<4, 4>(testvals);
    test_mul<3, 5>(testvals);
    test_mul<7, 2>(testvals);
    test_mul<kScalarTypeBits / 2 - 1, kScalarTypeBits / 2 - 2>(testvals);

    test_Rescale<0, 0>(testvals);
    test_Rescale<0, 1>(testvals);
    test_Rescale<2, 0>(testvals);
    test_Rescale<4, 4>(testvals);
    test_Rescale<4, 5>(testvals);
    test_Rescale<6, 3>(testvals);
    test_Rescale<13, 9>(testvals);

    test_ExactMulByPot<0, 0>(testvals);
    test_ExactMulByPot<0, 4>(testvals);
    test_ExactMulByPot<1, 4>(testvals);
    test_ExactMulByPot<3, 2>(testvals);
    test_ExactMulByPot<-4, 5>(testvals);
    test_ExactMulByPot<-2, 6>(testvals);

    fprintf(stderr, "PASS (%s)\n", msg);
  }
};

}  // end anonymous namespace

}  // end namespace gemmlowp

int main() {
  gemmlowp::TestFixedPoint<std::int32_t>().RunTests("Scalar int32");
  gemmlowp::TestFixedPoint<std::int16_t>().RunTests("Scalar int16");
#ifdef GEMMLOWP_SSE4
  gemmlowp::TestFixedPoint<__m128i>().RunTests("SSE4 __m128i = int32x4");
  gemmlowp::TestFixedPoint<gemmlowp::int16x8_m128i>().RunTests(
      "SSE4 __m128i = int16x8");
#endif
#ifdef GEMMLOWP_NEON
  gemmlowp::TestFixedPoint<int32x4_t>().RunTests("NEON int32x4_t");
  gemmlowp::TestFixedPoint<int16x8_t>().RunTests("NEON int16x8_t");
#endif
#ifdef GEMMLOWP_MSA
  gemmlowp::TestFixedPoint<v4i32>().RunTests("MSA v4i32");
  gemmlowp::TestFixedPoint<v8i16>().RunTests("MSA v8i16");
#endif
}
