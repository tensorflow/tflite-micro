/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_BINARY_FUNCTION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_BINARY_FUNCTION_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

template <typename T>
inline void CheckBinaryArithmeticParams(const ArithmeticParams& params) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  if(std::is_same<T,int8_t>::value) {
    // Input offset is negative input zero point. Activation tensors are
    // asymmetric quantized so they span the full int8 range.
    TFLITE_DCHECK_GE(-params.input1_offset, std::numeric_limits<int8_t>::min());
    TFLITE_DCHECK_GE(-params.input2_offset, std::numeric_limits<int8_t>::min());
    TFLITE_DCHECK_LE(-params.input1_offset, std::numeric_limits<int8_t>::max());
    TFLITE_DCHECK_LE(-params.input2_offset, std::numeric_limits<int8_t>::max());
 }
 else if(std::is_same<T,int16_t>::value) {
    TFLITE_DCHECK_EQ(params.input1_offset, 0);
    TFLITE_DCHECK_EQ(params.input2_offset, 0);
 }
}

template <typename T>
inline void BroadcastBinaryFunction4DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape,
    T* output_data,
    void (*check_arithmetic_params)(const ArithmeticParams&),
    T (*binary_func)(T, T, const ArithmeticParams&)) {
  check_arithmetic_params(params);
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] = binary_func(
              input1_data[SubscriptToIndex(desc1, b, y, x, c)],
              input2_data[SubscriptToIndex(desc2, b, y, x, c)], params);
        }
      }
    }
  }
}

template <typename T>
inline void BinaryFunction(const ArithmeticParams& params,
                           const RuntimeShape& input1_shape, const T* input1_data,
                           const RuntimeShape& input2_shape, const T* input2_data,
                           const RuntimeShape& output_shape, T* output_data,
                           void (*check_arithmetic_params)(const ArithmeticParams&),
                           T (*binary_func)(T, T, const ArithmeticParams&)) {
  check_arithmetic_params(params);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = binary_func(input1_data[i], input2_data[i], params);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_BINARY_FUNCTION_H_
