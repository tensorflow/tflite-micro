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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BINARY_FUNCTION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BINARY_FUNCTION_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Also appears to duplicate MinimumMaximum.
//
// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BroadcastBinaryFunction4DSlow(
    const RuntimeShape& unextended_input1_shape, const T1* input1_data,
    const RuntimeShape& unextended_input2_shape, const T2* input2_data,
    const RuntimeShape& unextended_output_shape, R* output_data,
    R (*func)(T1, T2)) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  const int* dims_data =
      reinterpret_cast<const int*>(output_shape.DimsDataUpTo5D());
  for (int b = 0; b < output_shape.Dims(0); ++b) {
    int out_idx_b = b * dims_data[1];
    int in_idx1_b = desc1.strides[0] * b;
    int in_idx2_b = desc2.strides[0] * b;
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      int out_idx_y = (out_idx_b + y) * dims_data[2];
      int in_idx1_y = in_idx1_b + desc1.strides[1] * y;
      int in_idx2_y = in_idx2_b + desc2.strides[1] * y;
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        int out_idx_x = (out_idx_y + x) * dims_data[3];
        int in1_idx = in_idx1_y + desc1.strides[2] * x;
        int in2_idx = in_idx2_y + desc2.strides[2] * x;
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = out_idx_x + c;
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = func(in1_val, in2_val);
          in1_idx += desc1.strides[3];
          in2_idx += desc2.strides[3];
        }
      }
    }
  }
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BinaryFunction(const RuntimeShape& input1_shape,
                           const T1* input1_data,
                           const RuntimeShape& input2_shape,
                           const T2* input2_data,
                           const RuntimeShape& output_shape, R* output_data,
                           R (*func)(T1, T2)) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = func(input1_data[i], input2_data[i]);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BINARY_FUNCTION_H_
