/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_context.h"

#include <cstdarg>
#include <cstddef>

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

int GetTensorIndex(int index, int max_size, const int* tensor_indices) {
  if (index >= 0 && index < max_size) {
    const int tensor_index = tensor_indices[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

}  // namespace

TfLiteTensor* MicroContext::AllocateTempInputTensor(const TfLiteNode* node,
                                                    int index) {
  const int tensor_index =
      GetTensorIndex(index, node->inputs->size, node->inputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempOutputTensor(const TfLiteNode* node,
                                                     int index) {
  const int tensor_index =
      GetTensorIndex(index, node->outputs->size, node->outputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempIntermediateTensor(
    const TfLiteNode* node, int index) {
  const int tensor_index = GetTensorIndex(index, node->intermediates->size,
                                          node->intermediates->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

void MicroContextReportOpError(struct TfLiteContext* context,
                               const char* format, ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

}  // namespace tflite
