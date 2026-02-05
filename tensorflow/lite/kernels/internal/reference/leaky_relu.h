/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

// Single-rounding MultiplyByQuantizedMultiplier
#if TFLITE_SINGLE_ROUNDING
/* Commented the condition check for quantized_multiplier to allow -ve values */
inline int32_t MultiplyByQuantizedMultiplier_v2(int32_t x, int32_t quantized_multiplier,
                                      int shift) {
  //TFLITE_DCHECK(quantized_multiplier >= 0);
  TFLITE_DCHECK(shift >= -31 && shift <= 30);

  const int64_t total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  int64_t result = x * static_cast<int64_t>(quantized_multiplier) + round;
  result = result >> total_shift;

  TFLITE_DCHECK(result >= std::numeric_limits<int32_t>::min() &&
                result <= std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(result);
}

/* Commented the condition check for quantized_multiplier to allow -ve values */
inline int32_t MultiplyByQuantizedMultiplier_v2(int64_t x, int32_t quantized_multiplier,
                                      int shift) {
  // Inputs:
  // - quantized_multiplier has fixed point at bit 31
  // - shift is -31 to +7 (negative for right shift)
  //
  // Assumptions: The following input ranges are assumed
  // - quantize_scale>=0  (the usual range is (1<<30) to (1>>31)-1)
  // - scaling is chosen so final scaled result fits in int32_t
  // - input x is in the range -(1<<47) <= x < (1<<47)
  //TFLITE_DCHECK(quantized_multiplier >= 0);
  TFLITE_DCHECK(shift >= -31 && shift < 8);
  TFLITE_DCHECK(x >= -(static_cast<int64_t>(1) << 47) &&
                x < (static_cast<int64_t>(1) << 47));

  const int32_t reduced_multiplier =
      (quantized_multiplier < 0x7FFF0000)
          ? ((quantized_multiplier + (1 << 15)) >> 16)
          : 0x7FFF;
  const int64_t total_shift = 15 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  int64_t result = x * static_cast<int64_t>(reduced_multiplier) + round;
  result = result >> total_shift;

  TFLITE_DCHECK(result >= std::numeric_limits<int32_t>::min() &&
                result <= std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(result);
}
// Double-rounding MultiplyByQuantizedMultiplier
#else
/* Call original function as the check on quantized_multiplier is relaxed */
inline int32_t MultiplyByQuantizedMultiplier_v2(int32_t x, int32_t quantized_multiplier,
                                      int shift) {
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

/* Call original function as the check on quantized_multiplier is relaxed */
inline int32_t MultiplyByQuantizedMultiplier_v2(int64_t x, int32_t quantized_multiplier,
                                      int shift) {
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}
#endif  // TFLITE_SINGLE_ROUNDING

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();
  for (int i = 0; i < flat_size; ++i) {
    const int32_t input_value = input_data[i] - params.input_offset;
    int32_t unclamped_output;
    if (input_value >= 0) {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier_v2(
                             input_value, params.output_multiplier_identity,
                             params.output_shift_identity);
    } else {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier_v2(
                             input_value, params.output_multiplier_alpha,
                             params.output_shift_alpha);
    }
    const T clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_