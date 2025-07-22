/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#if defined(HIFIMINI)

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils_hifimini.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

void FullyConnectedEvalHifimini(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  const int accum_depth_iters = accum_depth / 2;

  ae_p24x2s offsets_input_24x2 = AE_MOVPA24(input_offset);
  ae_p24x2s offsets_filter_24x2 = AE_MOVPA24(filter_offset);
  ae_q56s output_offset_56 = AE_CVTQ48A32S(output_offset);
  ae_q56s output_activation_max_56 = AE_CVTQ48A32S(output_activation_max);
  ae_q56s output_activation_min_56 = AE_CVTQ48A32S(output_activation_min);

  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      // Load intrinsics advance pointer before loading so backoff data pointers
      // by two before loading:
      const int8_t* input_ptr = (input_data + b * accum_depth) - 2;
      const int8_t* filter_ptr = (filter_data + out_c * accum_depth) - 2;

      // Main accumulator register entry for loop:
      ae_q56s sum_56 = AE_ZEROQ56();

      for (int d = 0; d < accum_depth_iters; d++) {
        // Load the signed 8bit values into the PR register:
        ae_p24x2s input_24x2;
        ae_p24x2s filter_24x2;
        AE_LP8X2F_IU(input_24x2, input_ptr, 2);
        AE_LP8X2F_IU(filter_24x2, filter_ptr, 2);

        // Right shift the signed 8bit values to expand to signed 24bit values:
        input_24x2 = AE_P24X2S_SRAI(input_24x2, 16);
        filter_24x2 = AE_P24X2S_SRAI(filter_24x2, 16);

        // Add offsets to data values (24 bit aligned):
        input_24x2 = AE_P24S_ADDS_P24X2S(offsets_input_24x2, input_24x2);
        filter_24x2 = AE_P24S_ADDS_P24X2S(offsets_filter_24x2, filter_24x2);

        // 24x2 signed integer dual MAC w/ addition into 56bit accumulator (48
        // bit aligned):
        AE_MULAAP24S_HH_LL(sum_56, input_24x2, filter_24x2);
      }

      // Left shift to get back into 32bit space (right padded to 48bit):
      sum_56 = AE_Q56S_SLAI(sum_56, 16);

      // Add bias data if needed:
      if (bias_data) {
        ae_q56s bias_56 = AE_CVTQ48A32S(bias_data[out_c]);
        sum_56 = AE_ADDQ56(sum_56, bias_56);
      }

      // Shift left into 24bit space and place back on PR register:
      sum_56 = AE_Q56S_SLAI(sum_56, 8);
      ae_p24x2s sum_24x2 = AE_TRUNCP24Q48(sum_56);

      // MultiplyByQuantizedMultiplier returns a 48bit aligned value
      sum_56 = MultiplyByQuantizedMultiplier(sum_24x2, output_multiplier,
                                             output_shift);

      // Add output_offset and cap min/max values:
      sum_56 = AE_ADDQ56(sum_56, output_offset_56);
      sum_56 = AE_MINQ56S(sum_56, output_activation_max_56);
      sum_56 = AE_MAXQ56S(sum_56, output_activation_min_56);

      output_data[out_c + output_depth * b] =
          static_cast<int8_t>(AE_TRUNCA32Q48(sum_56));
    }
  }
}

}  // namespace tflite
#endif  // defined(HIFIMINI)
