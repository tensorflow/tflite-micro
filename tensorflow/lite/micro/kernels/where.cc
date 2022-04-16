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
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/where.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {

namespace {
constexpr int kInputConditionTensor = 0;
constexpr int kOutputTensor = 0;


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* cond_tensor = tflite::micro::GetEvalInput(context, node, kInputConditionTensor);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TfLiteIntArray* dims = cond_tensor->dims;
  if (dims->size == 0) {
    // Scalar tensors are not supported.
    TF_LITE_KERNEL_LOG(context, "Where op requires condition w/ rank > 0");
    return kTfLiteError;
  }

  switch (cond_tensor->type) {
    case kTfLiteBool:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<bool>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteFloat32:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<float>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt64:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt32:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<int32_t>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt8:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<int8_t>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteUInt8:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<uint8_t>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteUInt32:
      reference_ops::SelectTrueCoords(tflite::micro::GetTensorShape(cond_tensor),
                                      tflite::micro::GetTensorData<uint32_t>(cond_tensor),
                                      tflite::micro::GetTensorData<int64_t>(output));
      break;
    default:
      MicroPrintf("Condition tensor has unsupported type: '%s'.",
                         TfLiteTypeGetName(cond_tensor->type));
  }
  return kTfLiteOk;
}
}  // namespace where

TfLiteRegistration Register_WHERE() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
