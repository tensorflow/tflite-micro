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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/l2normalization.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/l2norm.h"

namespace tflite {

const int kL2NormInputTensor = 0;
const int kL2NormOutputTensor = 0;

TfLiteStatus L2NormPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
  L2NormalizationParams* data =
      static_cast<L2NormalizationParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kL2NormInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kL2NormOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);

  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 ||
                              output->type == kTfLiteUInt8 ||
                              output->type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    data->input_zero_point = input->params.zero_point;
  } else if (output->type == kTfLiteFloat32) {
    data->input_zero_point = 0;
  }

  // Our implementations don't currently support activations.
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);

  return kTfLiteOk;
}

}  // namespace tflite
