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

#include "tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h"

#include <algorithm>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kBlockShapeTensor = 1;
constexpr int kPaddingTensor = 2;
constexpr int kOutputTensor = 0;

// Currently, only 3D NHC and 4D NHWC input/output op_context are supported.
// In case of 3D input, it will be extended to 3D NHWC by adding W=1.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(b/149952582): Support arbitrary dimension in SpaceToBatchND.
const int kInputOutputMinDimensionNum = 3;
const int kInputOutputMaxDimensionNum = 4;

void* SpaceToBatchNDInit(TfLiteContext* context, const char* buffer,
                         size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(SpaceToBatchParams));
}

TfLiteStatus ReshapeOutputTensor(TfLiteContext* context, const TfLiteNode* node,
                                 const TfLiteTensor* input,
                                 const TfLiteTensor* block_shape,
                                 const TfLiteTensor* padding,
                                 TfLiteTensor* output) {
  TF_LITE_ENSURE(context, IsConstantOrPersistentTensor(block_shape));
  TF_LITE_ENSURE(context, IsConstantOrPersistentTensor(padding));
  const int32_t* block_shape_data = GetTensorData<int32_t>(block_shape);
  const int32_t* padding_data = GetTensorData<int32_t>(padding);

  TfLiteIntArray* input_dims = input->dims;
  int spatial_dims_num = input_dims->size - 2;
  // Block_shape should be a 1D tensor with dimension [spatial_dims_num].
  TF_LITE_ENSURE_EQ(context, NumDimensions(block_shape), 1);
  TF_LITE_ENSURE_EQ(context, block_shape->dims->data[0], spatial_dims_num);
  // Padding should be a 2D tensor with dimension [spatial_dims_num, 2].
  TF_LITE_ENSURE_EQ(context, NumDimensions(padding), 2);
  TF_LITE_ENSURE_EQ(context, padding->dims->data[0], spatial_dims_num);
  TF_LITE_ENSURE_EQ(context, padding->dims->data[1], 2);

  // copy from input tensor as per TfLite code
  TF_LITE_ENSURE_EQ(context, input_dims->size, output->dims->size);
  RuntimeShape output_shape = GetTensorShape(input);
  // keep a copy of the output tensor shape for later comparison
  RuntimeShape old_output_shape = GetTensorShape(output);

  // Ensures the input height and width (with padding) is a multiple of block
  // shape height and width.
  int output_batch_size = input_dims->data[0];
  for (int dim = 0; dim < spatial_dims_num; ++dim) {
    int final_dim_size = (input_dims->data[dim + 1] + padding_data[dim * 2] +
                          padding_data[dim * 2 + 1]);
    TF_LITE_ENSURE(context, block_shape_data[dim] != 0);
    TF_LITE_ENSURE_EQ(context, final_dim_size % block_shape_data[dim], 0);
    output_shape.SetDim(dim + 1, final_dim_size / block_shape_data[dim]);
    output_batch_size *= block_shape_data[dim];
  }
  output_shape.SetDim(0, output_batch_size);
  output_shape.SetDim(input_dims->size - 1,
                      input_dims->data[input_dims->size - 1]);

  // check if need to relocate output tensor dims
  if (output_shape == old_output_shape) {
    return kTfLiteOk;
  } else if (output_shape.FlatSize() > old_output_shape.FlatSize() &&
             output->data.data != nullptr) {
    MicroPrintf(
        "SPACE_TO_BATCH_ND: resizing flatbuffer tensor data is not supported");
    return kTfLiteError;
  }

  // set the output tensor dims from output_shape
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_STATUS(tflite::micro::CreateWritableTensorDimsWithCopy(
      context, output, output_eval));
  std::copy_n(output_shape.DimsData(), output_shape.DimensionsCount(),
              output->dims->data);

  return kTfLiteOk;
}

TfLiteStatus SpaceToBatchNDPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* block_shape =
      micro_context->AllocateTempInputTensor(node, kBlockShapeTensor);
  TF_LITE_ENSURE(context, block_shape != nullptr);
  TfLiteTensor* padding =
      micro_context->AllocateTempInputTensor(node, kPaddingTensor);
  TF_LITE_ENSURE(context, padding != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) >= kInputOutputMinDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(output) >= kInputOutputMinDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(input) <= kInputOutputMaxDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(output) <= kInputOutputMaxDimensionNum);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);

  TF_LITE_ENSURE(context, node->user_data != nullptr);
  SpaceToBatchParams& params =
      *(static_cast<SpaceToBatchParams*>(node->user_data));

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, input->params.scale == output->params.scale);
    TF_LITE_ENSURE(context,
                   input->params.zero_point == output->params.zero_point);
    params.output_offset = output->params.zero_point;
  } else {
    params.output_offset = 0;
  }

  TfLiteStatus status =
      ReshapeOutputTensor(context, node, input, block_shape, padding, output);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(block_shape);
  micro_context->DeallocateTempTfLiteTensor(padding);
  micro_context->DeallocateTempTfLiteTensor(output);

  return status;
}

TfLiteStatus SpaceToBatchNDEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const SpaceToBatchParams& params =
      *(static_cast<const SpaceToBatchParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* block_shape =
      tflite::micro::GetEvalInput(context, node, kBlockShapeTensor);
  const TfLiteEvalTensor* padding =
      tflite::micro::GetEvalInput(context, node, kPaddingTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      reference_ops::SpaceToBatchND(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(block_shape),
          tflite::micro::GetTensorData<int32_t>(block_shape),
          tflite::micro::GetTensorShape(padding),
          tflite::micro::GetTensorData<int32_t>(padding),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::SpaceToBatchND(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(block_shape),
          tflite::micro::GetTensorData<int32_t>(block_shape),
          tflite::micro::GetTensorShape(padding),
          tflite::micro::GetTensorData<int32_t>(padding),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace.

TFLMRegistration Register_SPACE_TO_BATCH_ND() {
  return tflite::micro::RegisterOp(SpaceToBatchNDInit, SpaceToBatchNDPrepare,
                                   SpaceToBatchNDEval);
}

}  // namespace tflite
