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

#include "tensorflow/lite/micro/kernels/neg.h"

#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

TfLiteStatus NegEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kNegInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kNegOutputTensor);
  switch (input->type) {
    // TODO(wangtz): handle for kTfLiteInt8
    case kTfLiteFloat32: {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
      const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
      int err = xa_nn_elm_neg_f32_f32(
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorData<float>(input), flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      NegEvalReference<float>(input, output);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_NEG() {
  return tflite::micro::RegisterOp(nullptr, NegPrepareReference, NegEval);
}

}  // namespace micro
}  // namespace tflite
