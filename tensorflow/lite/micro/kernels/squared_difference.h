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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_SQUARED_DIFFERENCE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_SQUARED_DIFFERENCE_H_

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
inline constexpr int kSquaredDifferenceInputTensor1 = 0;
inline constexpr int kSquaredDifferenceInputTensor2 = 1;
inline constexpr int kSquaredDifferenceOutputTensor = 0;

struct OpDataSquaredDifference {
  bool requires_broadcast;
  ArithmeticParams arithmetic_params;
};

void* SquaredDifferenceInitReference(TfLiteContext* context, const char* buffer,
                                     size_t length);

TfLiteStatus SquaredDifferencePrepareReference(TfLiteContext* context,
                                               TfLiteNode* node);

template <typename T>
void EvalQuantizedSquaredDifferenceReference(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataSquaredDifference* data, const TfLiteEvalTensor* input1,
    const TfLiteEvalTensor* input2, TfLiteEvalTensor* output);

template <typename T>
void EvalSquaredDifferenceReference(TfLiteContext* context, TfLiteNode* node,
                                    const OpDataSquaredDifference* data,
                                    const TfLiteEvalTensor* input1,
                                    const TfLiteEvalTensor* input2,
                                    TfLiteEvalTensor* output);

// Generic must define registration function.
TFLMRegistration Register_SQUARED_DIFFERENCE();

}  // namespace micro
using micro::OpDataSquaredDifference;
using micro::Register_SQUARED_DIFFERENCE;
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_SQUARED_DIFFERENCE_H_
