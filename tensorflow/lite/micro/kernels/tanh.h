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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_TANH_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_TANH_H_

#include <cstdint>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

inline constexpr int kTanhInputTensor = 0;
inline constexpr int kTanhOutputTensor = 0;

struct OpDataTanh {
  int32_t input_zero_point = 0;
  int32_t input_range_radius = 0;
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
};

TfLiteStatus CalculateArithmeticOpDataTanh(TfLiteContext* context,
                                           TfLiteNode* node, OpDataTanh* data);

TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node);

}  // namespace micro
using micro::kTanhInputTensor;
using micro::kTanhOutputTensor;
using micro::OpDataTanh;
}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TANH_H_
