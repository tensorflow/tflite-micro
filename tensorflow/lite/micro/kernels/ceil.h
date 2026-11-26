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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CEIL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CEIL_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

inline constexpr int kCeilInputTensor = 0;
inline constexpr int kCeilOutputTensor = 0;

TfLiteStatus CeilPrepareReference(TfLiteContext* context, TfLiteNode* node);

void CeilEvalReference(const TfLiteEvalTensor* input, TfLiteEvalTensor* output);

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CEIL_H_
