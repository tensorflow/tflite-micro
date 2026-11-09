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

#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/micro/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

#if defined(USE_HIFI_ACT_TIE)
#include <xtensa/tie/xt_hifi2.h>
#endif  // defined(USE_HIFI_ACT_TIE)

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
inline constexpr int kInt8LutEntries = 256;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

struct OpDataTanhXtensa {
  OpDataTanh reference_op_data;
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  int8_t* tanh_lut = nullptr;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)
};

void* XtensaTanhInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(OpDataTanhXtensa));
  if (data == nullptr) {
    return nullptr;
  }
  return new (data) OpDataTanhXtensa();
}

TfLiteStatus XtensaTanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataTanhXtensa* data = static_cast<OpDataTanhXtensa*>(node->user_data);

  TF_LITE_ENSURE_OK(context, CalculateArithmeticOpDataTanh(
                                 context, node, &data->reference_op_data));

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kTanhInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);

#if defined(USE_HIFI_ACT_TIE) &&                        \
    (defined(AE_TANH16X4) || defined(AE_TANH16X4X2)) && \
    (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
  if (input->type == kTfLiteInt16 &&
      data->reference_op_data.input_multiplier != 0) {
    // Hardware TIE tanh expects input scaled by 4096 rather than 3 * 4096.
    double multiplier = static_cast<double>(input->params.scale) * 4096.0;
    data->reference_op_data.input_left_shift = 0;
    while (multiplier <= 32767.0 / 2.0 &&
           data->reference_op_data.input_left_shift <= 30) {
      data->reference_op_data.input_left_shift++;
      multiplier = multiplier * 2.0;
    }
    data->reference_op_data.input_multiplier = static_cast<int32_t>(multiplier);
  }
#endif  // defined(USE_HIFI_ACT_TIE) && ...

  const TfLiteType input_type = input->type;
  micro_context->DeallocateTempTfLiteTensor(input);

  if (input_type == kTfLiteInt8) {
    data->tanh_lut = static_cast<int8_t*>(context->AllocatePersistentBuffer(
        context, kInt8LutEntries * sizeof(int8_t)));
    TF_LITE_ENSURE(context, data->tanh_lut != nullptr);
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_init_lut_asym8s_tanh(data->tanh_lut,
                                   data->reference_op_data.input_zero_point,
                                   data->reference_op_data.input_range_radius,
                                   data->reference_op_data.input_multiplier,
                                   data->reference_op_data.input_left_shift),
        0);
  } else {
    data->tanh_lut = nullptr;
  }
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

  return kTfLiteOk;
}

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTanhInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTanhOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataTanhXtensa& xtensa_data =
      *static_cast<const OpDataTanhXtensa*>(node->user_data);
  const OpDataTanh& data = xtensa_data.reference_op_data;

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Tanh(tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<float>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      const int flat_size =
          MatchingFlatSize(tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorShape(output));
      if (flat_size == 0) {
        return kTfLiteOk;
      }
      int err = xa_nn_vec_tanh_sym16s_sym16s(
          tflite::micro::GetTensorData<int16_t>(output),
          tflite::micro::GetTensorData<int16_t>(input), data.input_multiplier,
          data.input_left_shift, flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      reference_integer_ops::Tanh(
          data.input_multiplier, data.input_left_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      TFLITE_DCHECK(xtensa_data.tanh_lut != nullptr);
      const int flat_size =
          MatchingFlatSize(tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorShape(output));
      if (flat_size == 0) {
        return kTfLiteOk;
      }
      int err = xa_nn_vec_apply_lut_asym8s_asym8s(
          tflite::micro::GetTensorData<int8_t>(output),
          tflite::micro::GetTensorData<int8_t>(input), xtensa_data.tanh_lut,
          kInt8LutEntries, flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      reference_integer_ops::Tanh(
          data.input_zero_point, data.input_range_radius, data.input_multiplier,
          data.input_left_shift, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_TANH() {
  return tflite::micro::RegisterOp(XtensaTanhInit, XtensaTanhPrepare, TanhEval);
}

}  // namespace micro
}  // namespace tflite
