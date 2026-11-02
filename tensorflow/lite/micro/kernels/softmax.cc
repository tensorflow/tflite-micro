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

#include "tensorflow/lite/micro/kernels/softmax.h"

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {

void* SoftmaxBackendInit(TfLiteContext* context, const char* buffer,
                         size_t length) {
  return SoftmaxInit(context, buffer, length);
}

TfLiteStatus SoftmaxBackendPrepare(TfLiteContext* context, TfLiteNode* node) {
  return SoftmaxPrepare(context, node);
}

TfLiteStatus SoftmaxEvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);
  const SoftmaxParams& op_data =
      *static_cast<const SoftmaxParams*>(node->user_data);

  tflite::micro::reference_ops::Softmax(
      op_data, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);
  const SoftmaxParams& op_data =
      *static_cast<const SoftmaxParams*>(node->user_data);

  tflite::micro::reference_ops::Softmax(
      op_data, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalInt8To16(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);
  const SoftmaxParams& op_data =
      *static_cast<const SoftmaxParams*>(node->user_data);

  tflite::micro::reference_ops::Softmax(
      op_data, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);
  const SoftmaxParams& op_data =
      *static_cast<const SoftmaxParams*>(node->user_data);

  tflite::micro::reference_ops::SoftmaxInt16(
      op_data, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int16_t>(input),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

namespace {

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteFloat32:
      return SoftmaxEvalFloat32(context, node);
    case kTfLiteInt8:
      return (output->type == kTfLiteInt16) ? SoftmaxEvalInt8To16(context, node)
                                            : SoftmaxEvalInt8(context, node);
    case kTfLiteInt16:
      return SoftmaxEvalInt16(context, node);
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
}

}  // namespace

TFLMRegistration Register_SOFTMAX() {
  return tflite::micro::RegisterOp(SoftmaxInit, SoftmaxPrepare, SoftmaxEval);
}

}  // namespace micro
}  // namespace tflite
