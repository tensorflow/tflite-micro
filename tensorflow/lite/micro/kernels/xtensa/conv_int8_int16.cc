/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"

namespace tflite {
namespace {

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(VISION_P6)
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  if (op_data.can_optimize) {
    // optimized Xtensa code will unpack filter data if necessary
#if defined(HIFI4) || defined(HIFI5)
    return ConvEvalHifiInt8(context, node, params, op_data, input, filter, bias,
                            output);
#elif defined(VISION_P6)
    return ConvEvalVision(context, node, params, op_data, input, filter, bias,
                          output);
#endif  // defined(HIFI4) || defined(HIFI5)
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(VISION_P6)

  return ConvReferenceEvalInt8(context, node);
}

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4)
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  if (op_data.can_optimize) {
    return ConvEvalHifiInt16(context, node, params, op_data, input, filter,
                             bias, output);
  }
#endif  // defined(HIFI4)

  return ConvReferenceEvalInt16(context, node);
}

}  // namespace

TfLiteRegistration_V1 Register_CONV_2D_INT8() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa, EvalInt8);
}

TfLiteRegistration_V1 Register_CONV_2D_INT16() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa,
                                   EvalInt16);
}

}  // namespace tflite
