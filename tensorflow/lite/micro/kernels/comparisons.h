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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_COMPARISONS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_COMPARISONS_H_

#include <cstddef>

#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

inline constexpr int kComparisonsInputTensor1 = 0;
inline constexpr int kComparisonsInputTensor2 = 1;
inline constexpr int kComparisonsOutputTensor = 0;

struct OpDataComparisons {
  ComparisonParams params{};
  bool is_quantized = false;
};

void* ComparisonsInit(TfLiteContext* context, const char* buffer,
                      size_t length);

TfLiteStatus ComparisonsPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus EqualEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus NotEqualEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus GreaterEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus GreaterEqualEvalReference(TfLiteContext* context,
                                       TfLiteNode* node);
TfLiteStatus LessEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus LessEqualEvalReference(TfLiteContext* context, TfLiteNode* node);

TFLMRegistration Register_EQUAL();
TFLMRegistration Register_NOT_EQUAL();
TFLMRegistration Register_GREATER();
TFLMRegistration Register_GREATER_EQUAL();
TFLMRegistration Register_LESS();
TFLMRegistration Register_LESS_EQUAL();

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_COMPARISONS_H_
