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

#include "tensorflow/lite/micro/kernels/elementwise.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
TfLiteStatus EvalLogicalNotHiFi(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kElementwiseInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kElementwiseOutputTensor);
  const size_t num_elements = ElementCount(*input->dims);

  const int8_t* input_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

  int err = xa_nn_elm_logicalnot_bool_bool(output_data_ptr, input_data_ptr,
                                           num_elements);
  TF_LITE_ENSURE(context, err == 0);
  return kTfLiteOk;
}
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
TfLiteStatus EvalNumericHiFi(TfLiteContext* context, TfLiteNode* node,
                             WORD32 (*xa_func)(FLOAT32*, const FLOAT32*,
                                               WORD32)) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kElementwiseInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kElementwiseOutputTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  const size_t num_elements = ElementCount(*input->dims);

  const float* in_data = tflite::micro::GetTensorData<float>(input);
  float* out_data = tflite::micro::GetTensorData<float>(output);

  int err = xa_func(out_data, in_data, num_elements);
  TF_LITE_ENSURE(context, err == 0);
  return kTfLiteOk;
}
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  const auto* op_data = static_cast<const OpDataAbsRsqrt*>(node->user_data);
  if (op_data->input_type == kTfLiteFloat32) {
    return EvalNumericHiFi(context, node, xa_nn_elm_abs_f32_f32);
  }
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return AbsEvalReference(context, node);
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return EvalNumericHiFi(context, node, xa_nn_elm_sine_f32_f32);
#else
  return SinEvalReference(context, node);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return EvalNumericHiFi(context, node, xa_nn_elm_cosine_f32_f32);
#else
  return CosEvalReference(context, node);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return EvalNumericHiFi(context, node, xa_nn_elm_logn_f32_f32);
#else
  return LogEvalReference(context, node);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return EvalNumericHiFi(context, node, xa_nn_elm_sqrt_f32_f32);
#else
  return SqrtEvalReference(context, node);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  const auto* op_data = static_cast<const OpDataAbsRsqrt*>(node->user_data);
  if (op_data->input_type == kTfLiteFloat32) {
    return EvalNumericHiFi(context, node, xa_nn_elm_rsqrt_f32_f32);
  }
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return RsqrtEvalReference(context, node);
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
  return EvalNumericHiFi(context, node, xa_nn_elm_square_f32_f32);
#else
  return SquareEvalReference(context, node);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return EvalLogicalNotHiFi(context, node);
#else
  return LogicalNotEvalReference(context, node);
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
}

}  // namespace

TFLMRegistration Register_ABS() {
  return tflite::micro::RegisterOp(ElementWiseAbsRsqrtInit,
                                   ElementWiseAbsPrepare, AbsEval);
}

TFLMRegistration Register_SIN() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare, SinEval);
}

TFLMRegistration Register_COS() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare, CosEval);
}

TFLMRegistration Register_LOG() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare, LogEval);
}

TFLMRegistration Register_SQRT() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   SqrtEval);
}

TFLMRegistration Register_RSQRT() {
  return tflite::micro::RegisterOp(ElementWiseAbsRsqrtInit,
                                   ElementWiseRsqrtPrepare, RsqrtEval);
}

TFLMRegistration Register_SQUARE() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   SquareEval);
}

TFLMRegistration Register_LOGICAL_NOT() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseLogicalPrepare,
                                   LogicalNotEval);
}

}  // namespace micro
}  // namespace tflite
