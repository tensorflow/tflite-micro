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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_common.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/logistic.h"
#include "tensorflow/lite/micro/micro_log.h"

#ifdef MCPS_MEASUREMENT
#include "tensorflow/lite/micro/kernels/ceva/mcps_macros.h"
#endif

namespace tflite {
namespace {

void* LogisticInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataLogistic));
}

TfLiteStatus LogisticEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kLogisticInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kLogisticOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataLogistic* data = static_cast<OpDataLogistic*>(node->user_data);

  if (input->type == kTfLiteFloat32) {
    switch (output->type) {
      case kTfLiteFloat32: {
#if defined(CEVA_BX1) || defined(CEVA_SP500)

        const float* input_data = tflite::micro::GetTensorData<float>(input);
        float* output_data = tflite::micro::GetTensorData<float>(output);
        const int flat_size =
            MatchingFlatSize(tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorShape(output));
#ifdef MCPS_MEASUREMENT
        MCPS_START_ONE;
#endif
        CEVA_TFLM_Logistic_float32(input_data, output_data, flat_size);
#ifdef MCPS_MEASUREMENT
        MCPS_STOP_ONE("Test params:CEVA_TFLM_Logistic_float32 loop = %d",
                      flat_size);
#endif

#else
        reference_ops::Logistic(tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<float>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
#endif  // ceva platform
        return kTfLiteOk;
      }
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt8) {
    switch (output->type) {
      case kTfLiteInt8: {
#if defined(CEVA_BX1) || defined(CEVA_SP500)
        int32_t input_zero_point = data->input_zero_point;
        int32_t input_range_radius = data->input_range_radius;
        int32_t input_multiplier = data->input_multiplier;
        int32_t input_left_shift = data->input_left_shift;
        int32_t input_size = NumElements(input->dims);
        const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
        int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

#ifdef MCPS_MEASUREMENT
        MCPS_START_ONE;
#endif
        CEVA_TFLM_Logistic_Int8(input_zero_point, input_range_radius,
                                input_multiplier, input_left_shift, input_size,
                                input_data, output_data);
#ifdef MCPS_MEASUREMENT
        MCPS_STOP_ONE("Test params:CEVA_TFLM_Logistic_Int8 loop = %d",
                      input_size);
#endif
#else
        reference_integer_ops::Logistic(
            data->input_zero_point, data->input_range_radius,
            data->input_multiplier, data->input_left_shift,
            NumElements(input->dims),
            tflite::micro::GetTensorData<int8_t>(input),
            tflite::micro::GetTensorData<int8_t>(output));
#endif  // ceva platform
        return kTfLiteOk;
      }
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else {
    MicroPrintf("Input %s, output %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_LOGISTIC() {
  return tflite::micro::RegisterOp(LogisticInit, LogisticPrepare, LogisticEval);
}
}  // namespace tflite
