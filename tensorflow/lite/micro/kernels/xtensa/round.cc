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

#include "tensorflow/lite/micro/kernels/round.h"

#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
namespace {

TfLiteStatus RoundEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kRoundInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kRoundOutputTensor);

#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
  int err = xa_nn_elm_round_f32_f32(tflite::micro::GetTensorData<float>(output),
                                    tflite::micro::GetTensorData<float>(input),
                                    flat_size);
  if (err != 0) {
    return kTfLiteError;
  }
#else
  RoundEvalReference(input, output);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_ROUND() {
  return tflite::micro::RegisterOp(nullptr, RoundPrepare, RoundEval);
}

}  // namespace micro
}  // namespace tflite
