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

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/elementwise.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

bool IsNumericSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32;
}

bool IsLogicalSupportedType(const TfLiteType type) {
  return type == kTfLiteBool;
}

typedef bool (*IsSupportedType)(TfLiteType);

TfLiteStatus GenericPrepareLogical(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!IsLogicalSupportedType(input->type)) {
    TF_LITE_KERNEL_LOG(context, "Input data type %s (%d) is not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GenericPrepareNumeric(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!IsNumericSupportedType(input->type)) {
    TF_LITE_KERNEL_LOG(context, "Input data type %s (%d) is not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             T func(T), TfLiteType expected_type) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const size_t num_elements = ElementCount(*input->dims);
  const T* in_data = tflite::micro::GetTensorData<T>(input);
  T* out_data = tflite::micro::GetTensorData<T>(output);
  for (size_t i = 0; i < num_elements; ++i) {
    out_data[i] = func(in_data[i]);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float)) {
  return EvalImpl<float>(context, node, float_func, kTfLiteFloat32);
}

TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool)) {
  return EvalImpl<bool>(context, node, bool_func, kTfLiteBool);
}

}  // namespace tflite
