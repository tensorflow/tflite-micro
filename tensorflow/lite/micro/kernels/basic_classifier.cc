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

#include <cstdint>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kTargetIndicesTensor = 1;
constexpr int kThresholdsTensor = 2;
constexpr int kDetectedTensor = 0;
constexpr int kTargetPosteriorsTensor = 1;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* target_indices =
      micro_context->AllocateTempInputTensor(node, kTargetIndicesTensor);
  TF_LITE_ENSURE(context, target_indices != nullptr);
  TfLiteTensor* thresholds =
      micro_context->AllocateTempInputTensor(node, kThresholdsTensor);
  TF_LITE_ENSURE(context, thresholds != nullptr);
  TfLiteTensor* detected =
      micro_context->AllocateTempOutputTensor(node, kDetectedTensor);
  TF_LITE_ENSURE(context, detected != nullptr);
  TfLiteTensor* target_posteriors =
      micro_context->AllocateTempOutputTensor(node, kTargetPosteriorsTensor);
  TF_LITE_ENSURE(context, target_posteriors != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(target_indices), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(thresholds), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(detected), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(target_posteriors), 1);

  const int num_targets = target_indices->dims->data[0];
  TF_LITE_ENSURE_EQ(context, thresholds->dims->data[0], num_targets);
  TF_LITE_ENSURE_EQ(context, detected->dims->data[0], num_targets);
  TF_LITE_ENSURE_EQ(context, target_posteriors->dims->data[0], num_targets);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, target_indices->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, thresholds->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, detected->type, kTfLiteBool);
  TF_LITE_ENSURE_TYPES_EQ(context, target_posteriors->type, kTfLiteInt32);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(target_indices);
  micro_context->DeallocateTempTfLiteTensor(thresholds);
  micro_context->DeallocateTempTfLiteTensor(detected);
  micro_context->DeallocateTempTfLiteTensor(target_posteriors);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TFLITE_DCHECK(input != nullptr);
  const TfLiteEvalTensor* target_indices =
      tflite::micro::GetEvalInput(context, node, kTargetIndicesTensor);
  TFLITE_DCHECK(target_indices != nullptr);
  const TfLiteEvalTensor* thresholds =
      tflite::micro::GetEvalInput(context, node, kThresholdsTensor);
  TFLITE_DCHECK(thresholds != nullptr);
  TfLiteEvalTensor* detected =
      tflite::micro::GetEvalOutput(context, node, kDetectedTensor);
  TFLITE_DCHECK(detected != nullptr);
  TfLiteEvalTensor* target_posteriors =
      tflite::micro::GetEvalOutput(context, node, kTargetPosteriorsTensor);
  TFLITE_DCHECK(target_posteriors != nullptr);

  const int32_t* input_data = tflite::micro::GetTensorData<int32_t>(input);
  const int32_t* target_indices_data =
      tflite::micro::GetTensorData<int32_t>(target_indices);
  const int32_t* thresholds_data =
      tflite::micro::GetTensorData<int32_t>(thresholds);
  bool* detected_data = tflite::micro::GetTensorData<bool>(detected);
  int32_t* target_posteriors_data =
      tflite::micro::GetTensorData<int32_t>(target_posteriors);

  const int32_t input_size = input->dims->data[0];
  const int num_targets = detected->dims->data[0];
  for (int i = 0; i < num_targets; ++i) {
    const int32_t target_idx = target_indices_data[i];
    if (target_idx < 0 || target_idx >= input_size) {
      return kTfLiteError;
    }
    detected_data[i] = input_data[target_idx] >= thresholds_data[i];
    target_posteriors_data[i] = input_data[target_idx];
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_BASIC_CLASSIFIER() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
