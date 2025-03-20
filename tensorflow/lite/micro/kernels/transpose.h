/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {

constexpr int kTransposeInputTensor = 0;
constexpr int kTransposePermTensor = 1;
constexpr int kTransposeOutputTensor = 0;

struct TransposeContext {
  TransposeContext(TfLiteContext* context, TfLiteNode* node) {
    micro_context = GetMicroContext(context);
    input = micro_context->AllocateTempInputTensor(node, kTransposeInputTensor);
    perm = micro_context->AllocateTempInputTensor(node, kTransposePermTensor);
    output =
        micro_context->AllocateTempOutputTensor(node, kTransposeOutputTensor);
  }
  ~TransposeContext() {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(perm);
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  MicroContext* micro_context;
  TfLiteTensor* input;
  TfLiteTensor* perm;
  TfLiteTensor* output;
};

TfLiteStatus TransposePrepare(TfLiteContext* context, TfLiteNode* node);
TFLMRegistration Register_TRANSPOSE();

#if defined(CMSIS_NN)
TFLMRegistration Register_TRANSPOSE_INT8();
#else
inline TFLMRegistration Register_TRANSPOSE_INT8() {
  return Register_TRANSPOSE();
}
#endif

}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_H_
