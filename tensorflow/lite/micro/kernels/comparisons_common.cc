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
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/comparisons.h"

namespace tflite {

const int kComparisonsInputTensor1 = 0;
const int kComparisonsInputTensor2 = 1;
const int kComparisonsOutputTensor = 0;

TfLiteStatus ComparisonsPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataComparisons* data = static_cast<OpDataComparisons*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kComparisonsInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  const TfLiteTensor* input2 = GetInput(context, node, kComparisonsInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);

  if (input1->type == kTfLiteUInt8 || input1->type == kTfLiteInt8) {
    auto input1_offset = -input1->params.zero_point;
    auto input2_offset = -input2->params.zero_point;
    const int kLeftShift = 8;

    int32_t input1_multiplier;
    int input1_shift;
    QuantizeMultiplierSmallerThanOneExp(
        static_cast<double>(input1->params.scale), &input1_multiplier,
        &input1_shift);
    int32_t input2_multiplier;
    int input2_shift;
    QuantizeMultiplierSmallerThanOneExp(
        static_cast<double>(input2->params.scale), &input2_multiplier,
        &input2_shift);

    data->params.left_shift = kLeftShift;
    data->params.input1_offset = input1_offset;
    data->params.input1_multiplier = input1_multiplier;
    data->params.input1_shift = input1_shift;
    data->params.input2_offset = input2_offset;
    data->params.input2_multiplier = input2_multiplier;
    data->params.input2_shift = input2_shift;
  }

  return kTfLiteOk;
}

}  // namespace tflite
