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

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/elementwise.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::abs);
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sin);
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::cos);
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::log);
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sqrt);
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, [](float f) { return 1.f / std::sqrt(f); });
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, [](float f) { return f * f; });
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalLogical(context, node, [](bool v) { return !v; });
}

TfLiteRegistration Register_ABS() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/AbsEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_SIN() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/SinEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_COS() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/CosEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LOG() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/LogEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_SQRT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/SqrtEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_RSQRT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/RsqrtEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_SQUARE() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareNumeric,
          /*invoke=*/SquareEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LOGICAL_NOT() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/
          GenericPrepareLogical,
          /*invoke=*/LogicalNotEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
