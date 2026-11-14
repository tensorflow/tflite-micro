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

#include "tensorflow/lite/micro/kernels/comparisons.h"

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
using XaCompareInt8Kernel = WORD32 (*)(WORD8*, const WORD8*, WORD32, WORD32,
                                       WORD32, const WORD8*, WORD32, WORD32,
                                       WORD32, WORD32, WORD32);

TfLiteStatus EvalComparisonInt8NoBroadcastHiFi(TfLiteContext* context,
                                               TfLiteNode* node,
                                               const TfLiteEvalTensor* input1,
                                               const TfLiteEvalTensor* input2,
                                               XaCompareInt8Kernel xa_kernel) {
  static_assert(sizeof(bool) == sizeof(int8_t),
                "NNLib comparison kernels write 1-byte boolean outputs.");
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto* data = static_cast<const OpDataComparisons*>(node->user_data);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);
  if (!tflite::micro::HaveSameShapes(input1, output)) {
    return kTfLiteError;
  }

  const int flat_size = tflite::micro::GetTensorShape(input1).FlatSize();

  int err = xa_kernel(
      tflite::micro::GetTensorData<int8_t>(output),
      tflite::micro::GetTensorData<int8_t>(input1), data->params.input1_offset,
      data->params.input1_shift, data->params.input1_multiplier,
      tflite::micro::GetTensorData<int8_t>(input2), data->params.input2_offset,
      data->params.input2_shift, data->params.input2_multiplier,
      data->params.left_shift, flat_size);
  if (err != 0) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  if (input1->type == kTfLiteInt8 &&
      tflite::micro::HaveSameShapes(input1, input2)) {
    return EvalComparisonInt8NoBroadcastHiFi(context, node, input1, input2,
                                             xa_nn_elm_equal_asym8sxasym8s);
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return EqualEvalReference(context, node);
}

TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  if (input1->type == kTfLiteInt8 &&
      tflite::micro::HaveSameShapes(input1, input2)) {
    return EvalComparisonInt8NoBroadcastHiFi(context, node, input1, input2,
                                             xa_nn_elm_notequal_asym8sxasym8s);
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return NotEqualEvalReference(context, node);
}

TfLiteStatus GreaterEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  if (input1->type == kTfLiteInt8 &&
      tflite::micro::HaveSameShapes(input1, input2)) {
    return EvalComparisonInt8NoBroadcastHiFi(context, node, input1, input2,
                                             xa_nn_elm_greater_asym8sxasym8s);
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return GreaterEvalReference(context, node);
}

TfLiteStatus GreaterEqualEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  if (input1->type == kTfLiteInt8 &&
      tflite::micro::HaveSameShapes(input1, input2)) {
    return EvalComparisonInt8NoBroadcastHiFi(
        context, node, input1, input2, xa_nn_elm_greaterequal_asym8sxasym8s);
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return GreaterEqualEvalReference(context, node);
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  if (input1->type == kTfLiteInt8 &&
      tflite::micro::HaveSameShapes(input1, input2)) {
    return EvalComparisonInt8NoBroadcastHiFi(context, node, input1, input2,
                                             xa_nn_elm_less_asym8sxasym8s);
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return LessEvalReference(context, node);
}

TfLiteStatus LessEqualEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  if (input1->type == kTfLiteInt8 &&
      tflite::micro::HaveSameShapes(input1, input2)) {
    return EvalComparisonInt8NoBroadcastHiFi(context, node, input1, input2,
                                             xa_nn_elm_lessequal_asym8sxasym8s);
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return LessEqualEvalReference(context, node);
}

}  // namespace

TFLMRegistration Register_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   EqualEval);
}

TFLMRegistration Register_NOT_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   NotEqualEval);
}

TFLMRegistration Register_GREATER() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   GreaterEval);
}

TFLMRegistration Register_GREATER_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   GreaterEqualEval);
}

TFLMRegistration Register_LESS() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   LessEval);
}

TFLMRegistration Register_LESS_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   LessEqualEval);
}

}  // namespace micro
}  // namespace tflite
