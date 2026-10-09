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

#include "tensorflow/lite/kernels/internal/reference/neg.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  switch (input->type) {
    // TODO(wangtz): handle for kTfLiteInt8
    case kTfLiteFloat32: {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI5) || defined(HIFI4))
      const int flat_size = MatchingFlatSize(tflite::micro::GetTensorShape(input), tflite::micro::GetTensorShape(output));
      const float* in_data = tflite::micro::GetTensorData<float>(input);
      float* out_data = tflite::micro::GetTensorData<float>(output);

      int err;
      err = xa_nn_elm_neg_f32_f32(out_data,
                                  in_data,
                                  flat_size
                                 );
      TF_LITE_ENSURE(context, (err==0) );
#else
      reference_ops::Negate(tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<float>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<float>(output));
#endif // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI5) || defined(HIFI4))
    }break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_NEG() {
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval);
}

}  // namespace tflite
