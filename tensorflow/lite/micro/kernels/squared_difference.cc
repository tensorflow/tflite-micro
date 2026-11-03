/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/squared_difference.h"

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

TfLiteStatus SquaredDifferenceEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSquaredDifference* data =
      static_cast<const OpDataSquaredDifference*>(node->user_data);

  const TfLiteEvalTensor* input1 = tflite::micro::GetEvalInput(
      context, node, kSquaredDifferenceInputTensor1);
  const TfLiteEvalTensor* input2 = tflite::micro::GetEvalInput(
      context, node, kSquaredDifferenceInputTensor2);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
      context, node, kSquaredDifferenceOutputTensor);

  if (output->type == kTfLiteFloat32) {
    EvalSquaredDifferenceReference<float>(context, node, data, input1, input2,
                                          output);
  } else if (output->type == kTfLiteInt32) {
    EvalSquaredDifferenceReference<int32_t>(context, node, data, input1, input2,
                                            output);
  } else if (output->type == kTfLiteInt8) {
    EvalQuantizedSquaredDifferenceReference<int8_t>(context, node, data, input1,
                                                    input2, output);
  } else if (output->type == kTfLiteInt16) {
    EvalQuantizedSquaredDifferenceReference<int16_t>(context, node, data,
                                                     input1, input2, output);
  } else {
    MicroPrintf(
        "SquaredDifference only supports FLOAT32, INT32 , INT16 and INT8 now, "
        "got %d.",
        output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace

TFLMRegistration Register_SQUARED_DIFFERENCE() {
  return tflite::micro::RegisterOp(
      SquaredDifferenceInit, SquaredDifferencePrepare, SquaredDifferenceEval);
}

}  // namespace micro
}  // namespace tflite
