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

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reshape.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

namespace {

TfLiteStatus ReshapeOutput(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kReshapeInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  // The shape tensor is optional
  TfLiteTensor* new_shape =
      micro_context->AllocateTempInputTensor(node, kReshapeShapeTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kReshapeOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements in the same as the number of input elements.
  int num_input_elements = NumElements(input);

  int output_shape_size = 0;
  int* output_shape_data = nullptr;
  if (new_shape != nullptr && new_shape->dims->size > 0) {
    // use new shape tensor data
    TF_LITE_ENSURE_EQ(context, new_shape->type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, new_shape->dims->size, 1);
    output_shape_data = GetTensorData<int32_t>(new_shape);
    output_shape_size = new_shape->dims->data[0];
    TF_LITE_ENSURE_EQ(context, output_shape_size,
                      static_cast<int32_t>(output->dims->size));
  } else {
    // use output shape
    output_shape_size = output->dims->size;
    output_shape_data = output->dims->data;
  }

  if (NumInputs(node) == 1 &&  // Legacy scalar supported with params.
      output_shape_size == 1 && output_shape_data[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    output_shape_size = 0;
  }

  int num_output_elements = 1;
  int stretch_dim = -1;
  for (int i = 0; i < output_shape_size; ++i) {
    int value = output_shape_data[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
    } else {
      num_output_elements *= value;
    }
  }
  if (stretch_dim != -1 || output_shape_size == 0) {
    TfLiteEvalTensor* output_eval =
        tflite::micro::GetEvalOutput(context, node, kReshapeOutputTensor);
    TF_LITE_ENSURE_STATUS(tflite::micro::CreateWritableTensorDimsWithCopy(
        context, output, output_eval));
    if (stretch_dim != -1) {
      output->dims->data[stretch_dim] =
          num_input_elements / num_output_elements;
      num_output_elements *= output->dims->data[stretch_dim];
    }
    output->dims->size = output_shape_size;
  }

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);

  micro_context->DeallocateTempTfLiteTensor(input);
  if (new_shape != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(new_shape);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus PrepareReshapeReference(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE_EQ(context, ReshapeOutput(context, node), kTfLiteOk);
  return kTfLiteOk;
}

}  // namespace tflite
