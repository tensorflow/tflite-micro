/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/internal/common.h"

#include <cstdint>
#include <limits>

namespace tflite {
namespace micro {

// Note on TFLITE_NO_SANITIZE_INTEGER_OVERFLOW below:
//
// The 32-bit MultiplyByQuantizedMultiplier overloads do not intentionally wrap,
// so they deliberately do NOT use WrappingMul/WrappingAdd. Under their
// documented input contract (asserted below) the arithmetic does not overflow;
// the attribute only silences UBSan reports at the contract's boundaries. Using
// wrapping here would be wrong -- it would mask a contract violation (a real
// bug) instead of letting it surface.

// Single-rounding MultiplyByQuantizedMultiplier
#if TFLITE_SINGLE_ROUNDING
TFLITE_NO_SANITIZE_INTEGER_OVERFLOW
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier,
                                      int shift) {
  TFLITE_DCHECK(quantized_multiplier >= 0);
  TFLITE_DCHECK(shift >= -31 && shift <= 30);

  const int64_t total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  int64_t result = x * static_cast<int64_t>(quantized_multiplier) + round;
  result = result >> total_shift;

  TFLITE_DCHECK(result >= std::numeric_limits<int32_t>::min() &&
                result <= std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(result);
}
// Double-rounding MultiplyByQuantizedMultiplier
#else
TFLITE_NO_SANITIZE_INTEGER_OVERFLOW
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier,
                                      int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                 x * (1 << left_shift), quantized_multiplier),
                             right_shift);
}
#endif  // TFLITE_SINGLE_ROUNDING

int32_t MultiplyByQuantizedMultiplier(int64_t x, int32_t quantized_multiplier,
                                      int shift) {
  // Inputs:
  // - quantized_multiplier has fixed point at bit 31
  // - shift is in [-31, 30] (negative for right shift)
  TFLITE_DCHECK(quantized_multiplier >= 0);
  if (quantized_multiplier <= 0 || x == 0) {
    return 0;
  }

  const int32_t reduced_multiplier =
      (quantized_multiplier < 0x7FFF0000)
          ? ((quantized_multiplier + (1 << 15)) >> 16)
          : 0x7FFF;
  if (reduced_multiplier == 0) {
    return 0;
  }

  constexpr int64_t kInt32Min = std::numeric_limits<int32_t>::min();
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  auto clamp_int32 = [](int64_t val) -> int32_t {
    if (val <= std::numeric_limits<int32_t>::min()) {
      return std::numeric_limits<int32_t>::min();
    }
    if (val >= std::numeric_limits<int32_t>::max()) {
      return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(val);
  };

  const int total_shift = 15 - shift;
  if (total_shift <= 0) {
    const int left_shift = -total_shift;
    if (x > kInt32Max) return static_cast<int32_t>(kInt32Max);
    if (x < kInt32Min) return static_cast<int32_t>(kInt32Min);
    const int64_t prod = x * static_cast<int64_t>(reduced_multiplier);
    if (left_shift >= 32) {
      return (prod > 0) ? static_cast<int32_t>(kInt32Max)
                        : static_cast<int32_t>(kInt32Min);
    }
    if (prod > (kInt32Max >> left_shift)) {
      return static_cast<int32_t>(kInt32Max);
    }
    if (prod < -((-kInt32Min) >> left_shift)) {
      return static_cast<int32_t>(kInt32Min);
    }
    return static_cast<int32_t>(prod * (static_cast<int64_t>(1) << left_shift));
  }

  if (total_shift <= 47 && x >= -(static_cast<int64_t>(1) << 47) &&
      x <= (static_cast<int64_t>(1) << 47)) {
    const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
    const int64_t result =
        (x * static_cast<int64_t>(reduced_multiplier) + round) >> total_shift;
    return clamp_int32(result);
  }

  if (total_shift >= 80) {
    return 0;
  }

  // Overflow-safe exact 64x15-bit rounding right shift without __int128.
  const int64_t x_lo =
      static_cast<int64_t>(static_cast<uint64_t>(x) & 0xFFFFFFFFULL);
  const int64_t x_hi = (x - x_lo) >> 32;
  const int64_t p_lo = x_lo * static_cast<int64_t>(reduced_multiplier);
  const int64_t p_hi = x_hi * static_cast<int64_t>(reduced_multiplier);

  if (total_shift < 32) {
    const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
    const int64_t low_with_round = p_lo + round;
    const int64_t hi_adj = p_hi + (low_with_round >> 32);
    const uint32_t low32 = static_cast<uint32_t>(low_with_round & 0xFFFFFFFFLL);
    const int rem_shift = 32 - total_shift;
    const int64_t low_part = static_cast<int64_t>(low32 >> total_shift);
    if (hi_adj > (kInt32Max >> rem_shift)) {
      return static_cast<int32_t>(kInt32Max);
    }
    if (hi_adj < -((-kInt32Min) >> rem_shift) - 1) {
      return static_cast<int32_t>(kInt32Min);
    }
    return clamp_int32(hi_adj * (static_cast<int64_t>(1) << rem_shift) +
                       low_part);
  }

  if (total_shift == 32) {
    const int64_t low_with_round = p_lo + (static_cast<int64_t>(1) << 31);
    return clamp_int32(p_hi + (low_with_round >> 32));
  }

  const int shift_hi = total_shift - 32;
  const int64_t hi_sum = p_hi + (p_lo >> 32);
  const int64_t round_hi = static_cast<int64_t>(1) << (shift_hi - 1);
  return clamp_int32((hi_sum + round_hi) >> shift_hi);
}

}  // namespace micro
}  // namespace tflite
