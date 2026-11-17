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

#include "tensorflow/lite/micro/kernels/tanh.h"

#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

void* TanhInit(TfLiteContext* context, const char* buffer, size_t length) {
  void* data = context->AllocatePersistentBuffer(context, sizeof(OpDataTanh));
  if (data == nullptr) {
    return nullptr;
  }
  return new (data) OpDataTanh();
}

TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  OpDataTanh* data = static_cast<OpDataTanh*>(node->user_data);

  return CalculateArithmeticOpDataTanh(context, node, data);
}

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTanhInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTanhOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataTanh& data = *static_cast<const OpDataTanh*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      TanhEvalReference<float>(data, input, output);
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      TanhEvalReference<int16_t>(data, input, output);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      TanhEvalReference<int8_t>(data, input, output);
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
  return tflite::micro::RegisterOp(TanhInit, TanhPrepare, TanhEval);
}

}  // namespace micro
}  // namespace tflite
