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

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

constexpr int kMaxDimensions = 8;

namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

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
  TF_LITE_ENSURE(context, NumDimensions(input) <= 8);
  TF_LITE_ENSURE(context, NumDimensions(input) >= NumElements(axis));

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteInt16) {
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

template <typename T>
void ReverseImpl(int32_t* axes, int num_axes, const RuntimeShape& input_shape,
                 const T* input_data, T* output_data) {
  bool is_upper = (axes[num_axes - 1] == input_shape.DimensionsCount() - 1);
  bool is_lower = (axes[0] == 0);
  int rank = input_shape.DimensionsCount();
  if (is_upper && is_lower) {
    std::reverse_copy(input_data, input_data + input_shape.FlatSize(),
                      output_data);
    return;
  } else {
    int32_t min_dim = axes[0];
    int32_t max_dim = axes[num_axes - 1];
    int upper_size = 1;
    for (int i = 0; i < min_dim; ++i) {
      upper_size *= input_shape.Dims(i);
    }
    int lower_size = 1;
    for (int i = max_dim + 1; i < rank; ++i) {
      lower_size *= input_shape.Dims(i);
    }
    int middle_size = 1;
    for (int i = min_dim; i <= max_dim; ++i) {
      middle_size *= input_shape.Dims(i);
    }

    if (lower_size > 1) {
      for (int i = 0; i < upper_size; ++i) {
        for (int j = 0; j < middle_size; ++j) {
          T* src = (T*)input_data + (i * (middle_size) + j) * lower_size;
          T* dst = (T*)output_data +
                   (i * (middle_size) + (middle_size - j - 1)) * lower_size;
          memcpy(dst, src, lower_size * sizeof(T));
        }
      }
    } else {
      for (int i = 0; i < upper_size; ++i) {
        std::reverse_copy(input_data + i * (middle_size),
                          input_data + i * middle_size + middle_size,
                          output_data + i * (middle_size));
      }
    }
  }
}

TfLiteStatus ReverseV2Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis =
      micro::GetEvalInput(context, node, kAxisTensor);
  TfLiteEvalTensor* output = micro::GetEvalOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);
  const int num_axes = static_cast<int>(ElementCount(*axis->dims));
  TF_LITE_ENSURE(context, num_axes <= 8);

  int32_t axes_data[kMaxDimensions];
  std::memcpy(axes_data, axis->data.i32, sizeof(int32_t) * num_axes);
  const int rank = tflite::micro::GetTensorShape(input).DimensionsCount();
  for (int i = 0; i < num_axes; ++i) {
    if (axes_data[i] < 0) {
      axes_data[i] += rank;
    }
    TF_LITE_ENSURE(context, axes_data[i] >= 0 && axes_data[i] < rank);
  }
  std::stable_sort(axes_data, axes_data + num_axes);

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
      ReverseImpl<float>(axes_data, num_axes,
                         tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<float>(input),
                         tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt32:
      ReverseImpl<int32_t>(axes_data, num_axes,
                           tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<int32_t>(input),
                           tflite::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteInt16:
      ReverseImpl<int16_t>(axes_data, num_axes,
                           tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<int16_t>(input),
                           tflite::micro::GetTensorData<int16_t>(output));
      break;
    case kTfLiteInt8:
    case kTfLiteUInt8:
      ReverseImpl<uint8_t>(axes_data, num_axes,
                           tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<uint8_t>(input),
                           tflite::micro::GetTensorData<uint8_t>(output));
      break;
    default:
      MicroPrintf(
          "Reverse currently supports float32, int16, "
          "int8 and uint8 for output, got %d.",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_REVERSE_V2() {
  return tflite::micro::RegisterOp(nullptr, ReverseV2Prepare, ReverseV2Eval);
}

}  // namespace tflite
