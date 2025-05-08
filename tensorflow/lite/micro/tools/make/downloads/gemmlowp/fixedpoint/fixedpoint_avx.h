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

// fixedpoint_avx.h: optimized avx specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_AVX_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_AVX_H_

#include <smmintrin.h>
#include "fixedpoint.h"

namespace gemmlowp {

template <>
struct FixedPointRawTypeTraits<__m256i> {
  typedef std::int32_t ScalarRawType;
  static const int kLanes = 4;
};

template <>
inline __m256i BitAnd(__m256i a, __m256i b) {
  return _mm256_and_si256(a, b);
}

template <>
inline __m256i BitOr(__m256i a, __m256i b) {
  return _mm256_or_si256(a, b);
}

template <>
inline __m256i BitXor(__m256i a, __m256i b) {
  return _mm256_xor_si256(a, b);
}

template <>
inline __m256i BitNot(__m256i a) {
  return _mm256_andnot_si256(a, _mm256_set1_epi32(-1));
}

template <>
inline __m256i Add(__m256i a, __m256i b) {
  return _mm256_add_epi32(a, b);
}

template <>
inline __m256i Mul(__m256i a, __m256i b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
inline __m256i Sub(__m256i a, __m256i b) {
  return _mm256_sub_epi32(a, b);
}

template <>
inline __m256i Neg(__m256i a) {
  return _mm256_sign_epi32(a, _mm256_set1_epi32(-1));
}

template <>
inline __m256i ShiftLeft(__m256i a, int offset) {
  return _mm256_slli_epi32(a, offset);
}

template <>
inline __m256i ShiftRight(__m256i a, int offset) {
  return _mm256_srai_epi32(a, offset);
}

template <>
inline __m256i SelectUsingMask(__m256i if_mask, __m256i then_val,
                               __m256i else_val) {
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(else_val),
                                              _mm256_castsi256_ps(then_val),
                                              _mm256_castsi256_ps(if_mask)));
}

template <>
inline __m256i MaskIfEqual(__m256i a, __m256i b) {
  return _mm256_cmpeq_epi32(a, b);
}

template <>
inline __m256i MaskIfNotEqual(__m256i a, __m256i b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline __m256i MaskIfZero(__m256i a) {
  return MaskIfEqual(a, _mm256_set1_epi32(0));
}

template <>
inline __m256i MaskIfNonZero(__m256i a) {
  return MaskIfNotEqual(a, _mm256_set1_epi32(0));
}

template <>
inline __m256i MaskIfGreaterThan(__m256i a, __m256i b) {
  return _mm256_cmpgt_epi32(a, b);
}

template <>
inline __m256i MaskIfLessThan(__m256i a, __m256i b) {
  return _mm256_cmpgt_epi32(b, a);
}

template <>
inline __m256i MaskIfGreaterThanOrEqual(__m256i a, __m256i b) {
  return BitNot(MaskIfLessThan(a, b));
}

template <>
inline __m256i MaskIfLessThanOrEqual(__m256i a, __m256i b) {
  return BitNot(MaskIfGreaterThan(a, b));
}

/* Assumptions:
   - All and Any are used on masks.
   - masks are all_ones for true lanes, all_zeroes otherwise.
Hence, All means all 128bits set, and Any means any bit set.
*/

template <>
inline bool All(__m256i a) {
  return _mm256_testc_si256(a, a);
}

template <>
inline bool Any(__m256i a) {
  return BitNot(_mm256_testz_si256(a, a));
}

template <>
inline __m256i RoundingHalfSum(__m256i a, __m256i b) {
  /* __m256i round_bit_mask, a_over_2, b_over_2, round_bit, sum; */
  /* We divide the inputs before the add to avoid the overflow and costly test
   */
  /* of checking if an overflow occured on signed add */
  /* round_bit_mask = _mm_set1_epi32(1); */
  /* a_over_2 = _mm_srai_epi32(a, 1); */
  /* b_over_2 = _mm_srai_epi32(b, 1); */
  /* sum = Add(a_over_2, b_over_2); */
  /* round_bit = _mm_sign_epi32(BitAnd(BitOr(a,b), round_bit_mask), sum); */
  /* return Add(sum, round_bit); */

  /* Other possibility detecting overflow and xor the sign if an overflow
   * happened*/
  __m256i one, sign_bit_mask, sum, rounded_half_sum, overflow, result;
  one = _mm256_set1_epi32(1);
  sign_bit_mask = _mm256_set1_epi32(0x80000000);
  sum = Add(a, b);
  rounded_half_sum = _mm256_srai_epi32(Add(sum, one), 1);
  overflow =
      BitAnd(BitAnd(BitXor(a, rounded_half_sum), BitXor(b, rounded_half_sum)),
             sign_bit_mask);
  result = BitXor(rounded_half_sum, overflow);
  return result;
}

template <>
inline __m256i SaturatingRoundingDoublingHighMul(__m256i a, __m256i b) {
  __m256i min, saturation_mask, a0_a2, a1_a3, b0_b2, b1_b3;
  __m256i a0b0_a2b2, a1b1_a3b3, a0b0_a2b2_rounded, a1b1_a3b3_rounded;
  __m256i a0b0_a2b2_rounded_2x, a1b1_a3b3_rounded_2x, result;
  __m256i nudge;

  // saturation only happen if a == b == INT_MIN
  min = _mm256_set1_epi32(std::numeric_limits<std::int32_t>::min());
  saturation_mask = BitAnd(MaskIfEqual(a, b), MaskIfEqual(a, min));

  // a = a0 | a1 | a2 | a3
  // b = b0 | b1 | b2 | b3
  a0_a2 = a;
  a1_a3 = _mm256_srli_si256(a, 4);
  b0_b2 = b;
  b1_b3 = _mm256_srli_si256(b, 4);

  a0b0_a2b2 = _mm256_mul_epi32(a0_a2, b0_b2);
  a1b1_a3b3 = _mm256_mul_epi32(a1_a3, b1_b3);

  // do the rounding and take into account that it will be doubled
  nudge = _mm256_set1_epi64x(1 << 30);
  a0b0_a2b2_rounded = _mm256_add_epi64(a0b0_a2b2, nudge);
  a1b1_a3b3_rounded = _mm256_add_epi64(a1b1_a3b3, nudge);

  // do the doubling
  a0b0_a2b2_rounded_2x = _mm256_slli_epi64(a0b0_a2b2_rounded, 1);
  a1b1_a3b3_rounded_2x = _mm256_slli_epi64(a1b1_a3b3_rounded, 1);

  // get the high part of the products
  result = _mm256_blend_epi16(_mm256_srli_si256(a0b0_a2b2_rounded_2x, 4),
                              a1b1_a3b3_rounded_2x, 0xcc);

  // saturate those which overflowed
  return SelectUsingMask(saturation_mask, min, result);
}

template <>
inline __m256i Dup<__m256i>(std::int32_t x) {
  return _mm256_set1_epi32(x);
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_AVX_H_