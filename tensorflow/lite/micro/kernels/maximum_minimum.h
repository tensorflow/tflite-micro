/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

// This file has a reference implementation of TFMaximum/TFMinimum.
enum KernelType {
  kReference,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input1 = tflite::micro::GetEvalInput(context, node, kInputTensor1);
    input2 = tflite::micro::GetEvalInput(context, node, kInputTensor2);
    output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
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

template <typename data_type, typename op_type>
void TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                     const OpContext& op_context) {
  reference_ops::MaximumMinimumBroadcastSlow(
      tflite::micro::GetTensorShape(op_context.input1),
      tflite::micro::GetTensorData<data_type>(op_context.input1),
      tflite::micro::GetTensorShape(op_context.input2),
      tflite::micro::GetTensorData<data_type>(op_context.input2),
      tflite::micro::GetTensorShape(op_context.output),
      tflite::micro::GetTensorData<data_type>(op_context.output),
      op_type::template op<data_type>);
}

TFLMRegistration Register_MAXIMUM();

TFLMRegistration Register_MINIMUM();

#if defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8.
TFLMRegistration Register_MAXIMUM_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int8.
TFLMRegistration Register_MINIMUM_INT8();

#else
// Note that while this block gets used for both reference and optimized kernels
// that do not have any specialized implementations, the only goal here is to
// define fallback implementation that allow reference kernels to still be used
// from applications that call a more specific kernel variant.
inline TFLMRegistration Register_MAXIMUM_INT8() { return Register_MAXIMUM(); }

inline TFLMRegistration Register_MINIMUM_INT8() { return Register_MINIMUM(); }

#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_MAXIMUM_MINIMUM_H_
