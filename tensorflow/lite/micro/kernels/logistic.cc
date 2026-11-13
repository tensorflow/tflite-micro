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

#include "tensorflow/lite/micro/kernels/logistic.h"

#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/micro/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

void* LogisticInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(OpDataLogistic));
  if (data == nullptr) {
    return nullptr;
  }
  return new (data) OpDataLogistic();
}

TfLiteStatus LogisticEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kLogisticInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kLogisticOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataLogistic& data =
      *static_cast<const OpDataLogistic*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Logistic(tflite::micro::GetTensorShape(input),
                              tflite::micro::GetTensorData<float>(input),
                              tflite::micro::GetTensorShape(output),
                              tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      reference_integer_ops::Logistic(
          data.input_multiplier, data.input_left_shift,
          NumElements(input->dims),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      reference_integer_ops::Logistic(
          data.input_zero_point, data.input_range_radius, data.input_multiplier,
          data.input_left_shift, NumElements(input->dims),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    default: {
      // TODO(b/141211002): Also support other data types once we have supported
      // temporary tensors in TFLM.
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_LOGISTIC() {
  return tflite::micro::RegisterOp(LogisticInit, LogisticPrepare, LogisticEval);
}
}  // namespace micro
}  // namespace tflite
