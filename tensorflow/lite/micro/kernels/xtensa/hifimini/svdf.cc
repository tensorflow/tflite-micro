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

#if defined(HIFIMINI)
#include "tensorflow/lite/micro/kernels/svdf.h"

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/hifimini/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_svdf.h"

namespace tflite {

/**
 * This version of SVDF is specific to TFLite Micro. It contains only a full
 * integer receipe with optimizations for the Xtensa HiFiMini platform.
 *
 * Note: passing OpDataSvdf by value might seem like an oversight but it helps
 * reduce the latency. See b/155656675 for more details.
 */
TfLiteStatus EvalIntegerSvdfHifimini(
    TfLiteContext* context, TfLiteNode* node,
    const TfLiteEvalTensor* input_tensor,
    const TfLiteEvalTensor* weights_feature_tensor,
    const TfLiteEvalTensor* weights_time_tensor,
    const TfLiteEvalTensor* bias_tensor, const TfLiteSVDFParams* params,
    TfLiteEvalTensor* activation_state_tensor, TfLiteEvalTensor* output_tensor,
    OpDataSvdf data) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  int32_t* scratch_tensor = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_tensor_index));
  TFLITE_DCHECK(scratch_tensor != nullptr);
  int32_t* scratch_output_tensor = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_output_tensor_index));
  TFLITE_DCHECK(scratch_output_tensor != nullptr);

  // Shift states.
  int16_t* const state_ptr =
      tflite::micro::GetTensorData<int16_t>(activation_state_tensor);

  // Left shift the activation_state.
  {
    int16_t* new_state_start = state_ptr;
    const int16_t* old_state_start = state_ptr + 1;
    const int16_t* old_state_end = state_ptr + n_batch * n_filter * n_memory;
    while (old_state_start != old_state_end) {
      *new_state_start++ = *old_state_start++;
    }
  }

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Feature matmul.
  {
    const int8_t* input = tflite::micro::GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        tflite::micro::GetTensorData<int8_t>(weights_feature_tensor);
    int16_t* result_in_batch = state_ptr + (n_memory - 1);

    ae_q56s output_int16_max_56 = AE_CVTQ48A32S(INT16_MAX);
    ae_q56s output_int16_min_56 = AE_CVTQ48A32S(INT16_MIN);
    ae_p24x2s input_zp_24x2 = AE_MOVPA24(data.input_zero_point);

    for (int b = 0; b < n_batch; b++) {
      const int8_t* weight_feature_ptr = weight_feature - 2;

      for (int r = 0; r < n_filter; r++) {
        ae_q56s dot_prod_56 = AE_ZEROQ56();

        const int8_t* input_batch_ptr = input + b * n_input;
        const int8_t* offset_input_batch_ptr = input_batch_ptr - 2;

        int num_iters = n_input / 2;
        for (int c = 0; c < num_iters; c++) {
          // Load 2 sets of values:
          ae_p24x2s weight_feature_ptr_24x2;
          ae_p24x2s input_batch_ptr_24x2;
          AE_LP8X2F_IU(weight_feature_ptr_24x2, weight_feature_ptr, 2);
          AE_LP8X2F_IU(input_batch_ptr_24x2, offset_input_batch_ptr, 2);

          // Right shift the signed 8bit values to expand to signed 24bit
          // values:
          weight_feature_ptr_24x2 = AE_P24X2S_SRAI(weight_feature_ptr_24x2, 16);
          input_batch_ptr_24x2 = AE_P24X2S_SRAI(input_batch_ptr_24x2, 16);

          // First subtract input_zp from input_batch_ptr_24x2:
          input_batch_ptr_24x2 =
              AE_SUBSP24S(input_batch_ptr_24x2, input_zp_24x2);

          // Multiply accum:
          AE_MULAAP24S_HH_LL(dot_prod_56, weight_feature_ptr_24x2,
                             input_batch_ptr_24x2);
        }

        // Left shift 48bit value into 24bit space and place on the PR register:
        dot_prod_56 = AE_Q56S_SLAI(dot_prod_56, 24);
        ae_p24x2s dot_prod_24x2 = AE_TRUNCP24Q48(dot_prod_56);

        dot_prod_56 = MultiplyByQuantizedMultiplier(
            dot_prod_24x2, data.effective_scale_1_a, data.effective_scale_1_b);

        // Cap min/max and convert to int32_t:
        dot_prod_56 = AE_MAXQ56S(dot_prod_56, output_int16_min_56);
        dot_prod_56 = AE_MINQ56S(dot_prod_56, output_int16_max_56);
        // Truncate immediately since the QR register is already 32 bit aligned:
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod_56.
        *result_in_batch = AE_TRUNCA32Q48(dot_prod_56);
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Perform batched vector dot product:
      const int16_t* vector1_ptr =
          tflite::micro::GetTensorData<int16_t>(weights_time_tensor);
      const int16_t* vector2_ptr = state_ptr + b * n_memory * n_filter;

      const ae_p16x2s* offset_vector1 =
          reinterpret_cast<const ae_p16x2s*>(vector1_ptr - 2);
      const ae_p16x2s* offset_vector2 =
          reinterpret_cast<const ae_p16x2s*>(vector2_ptr - 2);

      for (int i = 0; i < n_filter; i++) {
        *scratch_ptr_batch = 0;

        ae_q56s sum_56 = AE_ZEROQ56();
        int num_iters = n_memory / 2;
        for (int j = 0; j < num_iters; j++) {
          ae_p24x2s vector1_24x2;
          ae_p24x2s vector2_24x2;
          AE_LP16X2F_IU(vector1_24x2, offset_vector1, 4);
          AE_LP16X2F_IU(vector2_24x2, offset_vector2, 4);
          AE_MULAAP24S_HH_LL(sum_56, vector1_24x2, vector2_24x2);
        }
        // Truncate directly since values are already 32bit aligned:
        *scratch_ptr_batch = AE_TRUNCA32Q48(sum_56);
        scratch_ptr_batch++;
      }
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Add bias.
    if (bias_tensor) {
      // Vector batch assign:
      const int32_t* bias_data =
          tflite::micro::GetTensorData<int32_t>(bias_tensor);
      for (int i = 0; i < n_batch; ++i) {
        int32_t* output_ptr = scratch_output_tensor + i * n_unit;
        const int32_t* bias_ptr = bias_data;
        for (int j = 0; j < n_unit; ++j) {
          *output_ptr++ = *bias_ptr++;
        }
      }
    } else {
      int32_t* output_ptr = scratch_output_tensor;
      for (int i = 0; i < n_batch * n_unit; ++i) {
        *output_ptr++ = 0;
      }
    }

    // Reduce.
    for (int b = 0; b < n_batch; ++b) {
      int32_t* output_temp_ptr = scratch_output_tensor + b * n_unit;
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Reduction sum vector
      for (int i = 0; i < n_unit; ++i) {
        for (int j = 0; j < n_rank; ++j) {
          output_temp_ptr[i] += *scratch_ptr_batch++;
        }
      }
    }

    // Rescale.
    ae_q56s output_int8_max_56 = AE_CVTQ48A32S(INT8_MAX);
    ae_q56s output_int8_min_56 = AE_CVTQ48A32S(INT8_MIN);
    ae_q56s output_zp_56 = AE_CVTQ48A32S(data.output_zero_point);
    for (int i = 0; i < n_batch * n_unit; ++i) {
      ae_q56s x_56 = MultiplyByQuantizedMultiplierResult48Bit(
          scratch_output_tensor[i], data.effective_scale_2_a,
          data.effective_scale_2_b);
      // Add output adjustment:
      x_56 = AE_ADDQ56(x_56, output_zp_56);
      // Cap min/max and convert to int32_t (already aligned to 32bit):
      x_56 = AE_MAXQ56S(x_56, output_int8_min_56);
      x_56 = AE_MINQ56S(x_56, output_int8_max_56);
      tflite::micro::GetTensorData<int8_t>(output_tensor)[i] =
          static_cast<int8_t>(AE_TRUNCA32Q48(x_56));
    }
  }
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(HIFIMINI)
