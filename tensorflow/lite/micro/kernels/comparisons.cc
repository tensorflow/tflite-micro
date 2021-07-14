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
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/comparisons.h"

namespace tflite {
namespace {

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataComparisons* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data);
      break;
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// TODO(renjieliu): Refactor the logic to avoid duplications.
TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataComparisons* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data);
      break;
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataComparisons* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::GreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataComparisons* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataComparisons* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::LessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::LessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::LessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataComparisons* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataComparisons));
}

TfLiteRegistration Register_EQUAL() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ComparisonsPrepare,
          /*invoke=*/EqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_NOT_EQUAL() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ComparisonsPrepare,
          /*invoke=*/NotEqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_GREATER() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ComparisonsPrepare,
          /*invoke=*/GreaterEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_GREATER_EQUAL() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ComparisonsPrepare,
          /*invoke=*/GreaterEqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LESS() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ComparisonsPrepare,
          /*invoke=*/LessEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LESS_EQUAL() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/ComparisonsPrepare,
          /*invoke=*/LessEqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
