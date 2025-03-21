/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/transpose.h"

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/transpose.h"

namespace tflite {
namespace {

TfLiteStatus TransposeEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* perm_tensor =
      tflite::micro::GetEvalInput(context, node, kTransposePermTensor);
  const int size = perm_tensor->dims->data[0];
  TF_LITE_ENSURE(context, size <= 4);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTransposeInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTransposeOutputTensor);
  const cmsis_nn_transpose_params transpose_params = {
      size, reinterpret_cast<const uint32_t*>(perm_tensor->data.i32)};
  cmsis_nn_dims input_dims = {
      tflite::micro::GetTensorShape(input).DimsData()[0],
      tflite::micro::GetTensorShape(input).DimsData()[1],
      tflite::micro::GetTensorShape(input).DimsData()[2],
      tflite::micro::GetTensorShape(input).DimsData()[3]};
  cmsis_nn_dims output_dims = {
      tflite::micro::GetTensorShape(output).DimsData()[0],
      tflite::micro::GetTensorShape(output).DimsData()[1],
      tflite::micro::GetTensorShape(output).DimsData()[2],
      tflite::micro::GetTensorShape(output).DimsData()[3]};

  TFLITE_DCHECK_EQ(
      arm_transpose_s8(tflite::micro::GetTensorData<int8_t>(input),
                       tflite::micro::GetTensorData<int8_t>(output),
                       &input_dims, &output_dims, &transpose_params),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

TfLiteStatus TransposeEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* perm_tensor =
      tflite::micro::GetEvalInput(context, node, kTransposePermTensor);
  const int32_t* perm_data = perm_tensor->data.i32;
  const int size = perm_tensor->dims->data[0];
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

  // Transpose kernel only does rearranging values not numeric evaluations
  // on each cell. It's safe to implement per size of scalar type and this
  // trick keeps the total code size in a reasonable range.
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTransposeInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTransposeOutputTensor);
  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8: {
      TransposeEvalInt8(context, node);
    } break;
    case kTfLiteInt16:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<int16_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<int16_t>(output));
      break;
    default:
      MicroPrintf(
          "Type %s is currently not supported by Transpose. "
          "Only float32, int8 and int16 is supported",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_TRANSPOSE() {
  return tflite::micro::RegisterOp(nullptr, TransposePrepare, TransposeEval);
}
TFLMRegistration Register_TRANSPOSE_INT8() {
  return tflite::micro::RegisterOp(nullptr, TransposePrepare,
                                   TransposeEvalInt8);
}

}  // namespace tflite
