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

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

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
      LogisticEvalReference<float>(data, input, output);
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      LogisticEvalReference<int16_t>(data, input, output);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      LogisticEvalReference<int8_t>(data, input, output);
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
  return tflite::micro::RegisterOp(LogisticInitReference,
                                   LogisticPrepareReference, LogisticEval);
}
}  // namespace micro
}  // namespace tflite
