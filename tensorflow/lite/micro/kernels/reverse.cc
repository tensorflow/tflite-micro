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
#include "tensorflow/lite/kernels/internal/reference/reverse.h"

#include <stdint.h>

#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kMaxDimensions = RuntimeShape::kMaxSmallSize;
constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

int comp(const void* a, const void* b) {
  const int* int_a = static_cast<const int*>(a);
  const int* int_b = static_cast<const int*>(b);

  return (*int_a - *int_b);
}

TfLiteStatus ReverseV2Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Ensure inputs and outputs exist.
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* axis =
      micro_context->AllocateTempInputTensor(node, kAxisTensor);
  TF_LITE_ENSURE(context, axis != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(axis), 1);
  TF_LITE_ENSURE(context, NumDimensions(input) <= kMaxDimensions);
  TF_LITE_ENSURE(context, NumDimensions(input) >= NumElements(axis));

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteInt16 && input->type != kTfLiteInt64 &&
      input->type != kTfLiteBool) {
    MicroPrintf("Type '%s' is not supported by reverse.",
                TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if (axis->type != kTfLiteInt32) {
    MicroPrintf("Axis Type '%s' is not supported by reverse.",
                TfLiteTypeGetName(axis->type));
    return kTfLiteError;
  }
  // The value type and output type must match.
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(axis);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus ReverseV2Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis =
      micro::GetEvalInput(context, node, kAxisTensor);
  TfLiteEvalTensor* output = micro::GetEvalOutput(context, node, kOutputTensor);

  const int num_axes = static_cast<int>(ElementCount(*axis->dims));

  // TFLite reverse implementation is expecting fixed size 8,
  // so using 8 below.
  std::array<int32_t, 8> axes_data;
  std::memcpy(axes_data.data(), axis->data.data, sizeof(int32_t) * num_axes);
  const int rank = tflite::micro::GetTensorShape(input).DimensionsCount();
  for (int i = 0; i < num_axes; ++i) {
    if (axes_data[i] < 0) {
      axes_data[i] += rank;
    }
    TF_LITE_ENSURE(context, axes_data[i] >= 0 && axes_data[i] < rank);
  }
  std::qsort(axes_data.data(), num_axes, sizeof(int32_t), comp);

  bool is_contiguous = true;
  for (int i = 1; i < num_axes; ++i) {
    if (axes_data[i - 1] + 1 != axes_data[i]) {
      is_contiguous = false;
      break;
    }
  }
  if (!is_contiguous) {
    MicroPrintf("Non-contiguous `axes` not supported");
    return kTfLiteError;
  }

  switch (output->type) {
    case kTfLiteFloat32:
      reference_ops::Reverse<float>(
          axes_data, num_axes, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt32:
      reference_ops::Reverse<int32_t>(
          axes_data, num_axes, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int32_t>(input),
          tflite::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::Reverse<int16_t>(
          axes_data, num_axes, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorData<int16_t>(output));
      break;
    case kTfLiteInt8:
    case kTfLiteUInt8:
      reference_ops::Reverse<uint8_t>(
          axes_data, num_axes, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<uint8_t>(input),
          tflite::micro::GetTensorData<uint8_t>(output));
      break;
    case kTfLiteInt64:
      reference_ops::Reverse<int64_t>(
          axes_data, num_axes, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int64_t>(input),
          tflite::micro::GetTensorData<int64_t>(output));
      break;
    case kTfLiteBool:
      reference_ops::Reverse<bool>(axes_data, num_axes,
                                   tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorData<bool>(input),
                                   tflite::micro::GetTensorData<bool>(output));
      break;
    default:
      MicroPrintf("Output type '%s' (%d) is not supported.",
                  TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_REVERSE_V2() {
  return tflite::micro::RegisterOp(nullptr, ReverseV2Prepare, ReverseV2Eval);
}

}  // namespace tflite
