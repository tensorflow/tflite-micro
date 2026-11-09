/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/tanh.h"

#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {
namespace {

TfLiteStatus CalculateArithmeticOpDataTanhImpl(TfLiteContext* context,
                                               const TfLiteTensor* input,
                                               const TfLiteTensor* output,
                                               OpDataTanh* data) {
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  data->input_zero_point = input->params.zero_point;

  if (input->type == kTfLiteInt8) {
    static constexpr int kInputIntegerBits = 4;
    const double input_real_multiplier =
        static_cast<double>(input->params.scale) *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    const double q = std::frexp(input_real_multiplier, &data->input_left_shift);
    data->input_multiplier = static_cast<int32_t>(TfLiteRound(q * (1ll << 31)));

    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 31);
  } else if (input->type == kTfLiteInt16) {
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;

    // These operators are implemented in fixed-point arithmetic,
    // which intrinsically wants symmetric ranges (zero_point==0)
    // and power-of-two scales (power-of-two is abbreviated below as POT).
    // While more general support would be possible by means of rescaling,
    // that would add some overhead and some loss of accuracy and wouldn't
    // be used at the moment as current quantized LSTM applications are
    // happy with symmetric, power-of-two-scales quantization. So we just
    // implement that narrow case only for now.

    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    int input_scale_log2_rounded;
    bool param_scale_pot =
        CheckedLog2(input->params.scale, &input_scale_log2_rounded);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    param_scale_pot &=
        (data->input_left_shift == 0 || data->input_left_shift == 1);

    if (param_scale_pot) {
      data->input_multiplier = 0;
    } else {
      // Calculate multiplier to change input scale to 1/(3*4096)
      // as required by the table lookup.
      // The number 3.0 in the multiplier comes from here,
      // because the interval is [-10.7, 10.7] instead of [-8, 8].
      // So, in this scaling +/-2^17 represents +/-10.7.

      double multiplier =
          static_cast<double>(input->params.scale) * 4096.0 * 3.0;
      data->input_left_shift = 0;

      while (multiplier <= 32767.0 / 2.0 && data->input_left_shift <= 30) {
        data->input_left_shift++;
        multiplier = multiplier * 2.0;
      }

      data->input_multiplier = static_cast<int32_t>(multiplier);
    }
    TFLITE_DCHECK_LE(data->input_multiplier, 32767);
    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);
  } else if (input->type != kTfLiteFloat32) {
    MicroPrintf("Input %s, output %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteStatus CalculateArithmeticOpDataTanh(TfLiteContext* context,
                                           TfLiteNode* node, OpDataTanh* data) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kTanhInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kTanhOutputTensor);
  if (output == nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
    return kTfLiteError;
  }

  TfLiteStatus status =
      CalculateArithmeticOpDataTanhImpl(context, input, output, data);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return status;
}

TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataTanh* data = static_cast<OpDataTanh*>(node->user_data);

  return CalculateArithmeticOpDataTanh(context, node, data);
}

}  // namespace micro
}  // namespace tflite
