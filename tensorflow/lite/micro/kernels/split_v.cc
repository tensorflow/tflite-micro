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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

template <typename T>
TfLiteStatus SplitImpl(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteEvalTensor* input, int axis_value) {
  const TfLiteIntArray* input_dims = input->dims;
  const TfLiteEvalTensor* output0 =
      tflite::micro::GetEvalOutput(context, node, 0);

  const int split_dimensions = input_dims->size;

  TFLITE_DCHECK_LT(axis_value, split_dimensions);
  TFLITE_DCHECK_EQ(output0->dims->size, split_dimensions);

  int64_t split_size = 0;
  const int output_count = NumOutputs(node);
  for (int i = 0; i < output_count; i++) {
    split_size +=
        tflite::micro::GetEvalOutput(context, node, i)->dims->data[axis_value];
  }
  TFLITE_DCHECK_EQ(split_size, input_dims->data[axis_value]);
  int64_t outer_size = 1;
  for (int i = 0; i < axis_value; ++i) {
    outer_size *= input_dims->data[i];
  }

  int64_t base_inner_size = 1;
  for (int i = axis_value + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_dims->data[i];
  }

  const T* input_ptr = tflite::micro::GetTensorData<T>(input);
  for (int k = 0; k < outer_size; ++k) {
    for (int i = 0; i < output_count; ++i) {
      TfLiteEvalTensor* output_tensor =
          tflite::micro::GetEvalOutput(context, node, i);
      T* output_data = tflite::micro::GetTensorData<T>(output_tensor);
      const int copy_size =
          output_tensor->dims->data[axis_value] * base_inner_size;
      T* output_ptr = output_data + k * copy_size;
      for (int j = 0; j < copy_size; ++j) output_ptr[j] = input_ptr[j];
      input_ptr += copy_size;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus SplitVPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);

  MicroContext* micro_context = GetMicroContext(context);
  // Dynamic output tensors are needed if axis tensor is not constant.
  // But Micro doesn't support dynamic memory allocation, so we only support
  // constant axis tensor for now.
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 2);
  TF_LITE_ENSURE_MSG(context, IsConstantTensor(axis),
                     "Non-constant >axis< tensor is not supported");
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus SplitVEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 2);

  int axis_value = tflite::micro::GetTensorData<int32_t>(axis)[0];
  if (axis_value < 0) {
    axis_value += input->dims->size;
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < input->dims->size);

  switch (input->type) {
    case kTfLiteFloat32: {
      return SplitImpl<float>(context, node, input, axis_value);
    }
    case kTfLiteInt8: {
      return SplitImpl<int8_t>(context, node, input, axis_value);
    }
    case kTfLiteInt16: {
      return SplitImpl<int16_t>(context, node, input, axis_value);
    }
    case kTfLiteInt32: {
      return SplitImpl<int32_t>(context, node, input, axis_value);
    }
    default:
      MicroPrintf("Type %s currently not supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_SPLIT_V() {
  return tflite::micro::RegisterOp(nullptr, SplitVPrepare, SplitVEval);
}

}  // namespace tflite
