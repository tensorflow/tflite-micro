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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_MAXIMUM_MINIMUM_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_MAXIMUM_MINIMUM_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

// This file has a reference implementation of TFMaximum/TFMinimum.
enum KernelType {
  kReference,
};

extern const int kMaximumMinimumInputTensor1;
extern const int kMaximumMinimumInputTensor2;
extern const int kMaximumMinimumOutputTensor;

struct OpContextMaximumMinimum {
  OpContextMaximumMinimum(TfLiteContext* context, TfLiteNode* node) {
    input1 = tflite::micro::GetEvalInput(context, node, kMaximumMinimumInputTensor1);
    input2 = tflite::micro::GetEvalInput(context, node, kMaximumMinimumInputTensor2);
    output = tflite::micro::GetEvalOutput(context, node, kMaximumMinimumOutputTensor);
  }
  const TfLiteEvalTensor* input1;
  const TfLiteEvalTensor* input2;
  TfLiteEvalTensor* output;
};

struct MaximumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 > el2 ? el1 : el2;
  }
};

struct MinimumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 < el2 ? el1 : el2;
  }
};

}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_MAXIMUM_MINIMUM_H_
