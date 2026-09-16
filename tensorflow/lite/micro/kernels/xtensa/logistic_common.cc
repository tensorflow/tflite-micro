/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/logistic.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_logistic.h"
#if defined(USE_HIFI_ACT_TIE)
#include <xtensa/tie/xt_hifi2.h>
#endif

namespace tflite {
const int kLogisticInputTensor = 0;
const int kLogisticOutputTensor = 0;

TfLiteStatus CalculateArithmeticOpDataLogistic(TfLiteContext* context,
                                               TfLiteNode* node,
                                               OpDataLogistic* data) {
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kLogisticInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kLogisticOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                      std::numeric_limits<int8_t>::min());

    static constexpr int kInputIntegerBits = 4;
    const double input_real_multiplier =
        static_cast<double>(input->params.scale) *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    data->input_zero_point = input->params.zero_point;

    const double q = std::frexp(input_real_multiplier, &data->input_left_shift);
    data->input_multiplier = static_cast<int32_t>(TfLiteRound(q * (1ll << 31)));

    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 31);
  }

  if (input->type == kTfLiteInt16) {
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;

    // See comments in TanhPrepare about requiring zero_point==0
    // and a power-of-two ("POT") scale.

    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    int input_scale_log2_rounded;
    bool param_scale_pot =
        CheckedLog2(input->params.scale, &input_scale_log2_rounded);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    param_scale_pot &= (data->input_left_shift == 0);

    if (param_scale_pot) {
      data->input_multiplier = 0;
    } else {
      // Calculate multiplier to change input scale to 1/(3*4096)
      // as required by the table lookup.
      // In this scaling +/-2^17 represents +/-10.7
#if (defined(USE_HIFI_ACT_TIE) && (defined(AE_SIGMOID16X4) || defined(AE_SIGMOID16X4X2)))
      double multiplier =
          static_cast<double>(input->params.scale) * 4096.0;
#else      
      double multiplier =
          static_cast<double>(input->params.scale) * 4096.0 * 3.0;
#endif
      data->input_left_shift = 0;

      while (multiplier <= 32767.0 / 2.0 && data->input_left_shift <= 30) {
        data->input_left_shift++;
        multiplier = multiplier * 2.0;
      }

      data->input_multiplier = static_cast<int32_t>(multiplier);
    }

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus LogisticPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* xtensa_data = static_cast<OpDataLogisticXtensa*>(node->user_data);
  TF_LITE_ENSURE_OK(context, CalculateArithmeticOpDataLogistic(
                                  context, node, &xtensa_data->reference_op_data));

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kLogisticInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  if (input->type == kTfLiteInt8)
  {
    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int8_t) * 256);
    TF_LITE_ENSURE(context, raw != nullptr);
    xtensa_data->sigmoid_lut = raw;
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_init_lut_asym8s_sigmoid(
            static_cast<int8_t*>(xtensa_data->sigmoid_lut),
            xtensa_data->reference_op_data.input_zero_point,
            xtensa_data->reference_op_data.input_range_radius,
            xtensa_data->reference_op_data.input_multiplier,
            xtensa_data->reference_op_data.input_left_shift),
        0);
  } else {
    xtensa_data->sigmoid_lut = nullptr;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return kTfLiteOk;
}

}  // namespace tflite
