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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_LOGISTIC_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_LOGISTIC_H_

#include "tensorflow/lite/micro/kernels/logistic.h"

namespace tflite {

struct OpDataLogisticXtensa {
  OpDataLogistic reference_op_data;
  // Points to a 256-entry int8_t sigmoid LUT for kTfLiteInt8 inputs;
  void* sigmoid_lut;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_LOGISTIC_H_
