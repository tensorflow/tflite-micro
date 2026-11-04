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
#include "tensorflow/lite/micro/kernels/squared_difference.h"

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
TfLiteStatus EvalQuantizedSquaredDifferenceInt8Hifi(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataSquaredDifference* data, const TfLiteEvalTensor* input1,
    const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  const ArithmeticParams& params = data->arithmetic_params;
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  err = xa_nn_elm_squared_diff_broadcast_4D_asym8sxasym8s_asym8s(
      tflite::micro::GetTensorData<int8_t>(output),
      extended_output_shape.DimsData(), params.output_offset,
      params.output_shift, params.output_multiplier,
      params.quantized_activation_min, params.quantized_activation_max,
      tflite::micro::GetTensorData<int8_t>(input1),
      extended_input1_shape.DimsData(), params.input1_offset,
      params.input1_shift, params.input1_multiplier,
      tflite::micro::GetTensorData<int8_t>(input2),
      extended_input2_shape.DimsData(), params.input2_offset,
      params.input2_shift, params.input2_multiplier, params.left_shift);
  TF_LITE_ENSURE(context, err == 0);
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedSquaredDifferenceInt16Hifi(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataSquaredDifference* data, const TfLiteEvalTensor* input1,
    const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  const ArithmeticParams& params = data->arithmetic_params;
  int err;
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input1));
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input2));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  err = xa_nn_elm_squared_diff_broadcast_4D_sym16sxsym16s_sym16s(
      tflite::micro::GetTensorData<int16_t>(output),
      extended_output_shape.DimsData(), params.output_shift,
      params.output_multiplier, params.quantized_activation_min,
      params.quantized_activation_max,
      tflite::micro::GetTensorData<int16_t>(input1),
      extended_input1_shape.DimsData(), params.input1_shift,
      params.input1_multiplier, tflite::micro::GetTensorData<int16_t>(input2),
      extended_input2_shape.DimsData(), params.input2_shift,
      params.input2_multiplier, params.left_shift);
  TF_LITE_ENSURE(context, err == 0);
  return kTfLiteOk;
}
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

TfLiteStatus SquaredDifferenceEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSquaredDifference* data =
      static_cast<const OpDataSquaredDifference*>(node->user_data);

  const TfLiteEvalTensor* input1 = tflite::micro::GetEvalInput(
      context, node, kSquaredDifferenceInputTensor1);
  const TfLiteEvalTensor* input2 = tflite::micro::GetEvalInput(
      context, node, kSquaredDifferenceInputTensor2);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
      context, node, kSquaredDifferenceOutputTensor);

  if (output->type == kTfLiteFloat32) {
    EvalSquaredDifferenceReference<float>(context, node, data, input1, input2,
                                          output);
  } else if (output->type == kTfLiteInt32) {
    EvalSquaredDifferenceReference<int32_t>(context, node, data, input1, input2,
                                            output);
  } else if (output->type == kTfLiteInt8) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
    TF_LITE_ENSURE_OK(context,
                      EvalQuantizedSquaredDifferenceInt8Hifi(
                          context, node, data, input1, input2, output));
#else
    EvalQuantizedSquaredDifferenceReference<int8_t>(context, node, data, input1,
                                                    input2, output);
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  } else if (output->type == kTfLiteInt16) {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
    TF_LITE_ENSURE_OK(context,
                      EvalQuantizedSquaredDifferenceInt16Hifi(
                          context, node, data, input1, input2, output));
#else
    EvalQuantizedSquaredDifferenceReference<int16_t>(context, node, data,
                                                     input1, input2, output);
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  } else {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(output->type),
                output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace

TFLMRegistration Register_SQUARED_DIFFERENCE() {
  return tflite::micro::RegisterOp(SquaredDifferenceInitReference,
                                   SquaredDifferencePrepareReference,
                                   SquaredDifferenceEval);
}

}  // namespace micro
}  // namespace tflite
