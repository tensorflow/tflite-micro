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
#include <cstdint>

#include "tensorflow/lite/micro/kernels/batch_to_space_nd.h"
#include "tensorflow/lite/micro/kernels/internal/reference/batch_to_space_nd.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {

// Currently, only 3D NHC and 4D NHWC input/output tensors are supported.
// In case of 3D input, it will be extended to 4D NHWC by adding W=1.
// The 4D array needs to have exactly 2 spatial dimensions.
// TODO(b/149952582): Support arbitrary dimension in SpaceToBatchND.
TfLiteStatus BatchToSpaceNDPrepareReference(TfLiteContext* context,
                                            TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kBatchToSpaceNDInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* block_shape = micro_context->AllocateTempInputTensor(
      node, kBatchToSpaceNDBlockShapeTensor);
  TF_LITE_ENSURE(context, block_shape != nullptr);
  TfLiteTensor* crops =
      micro_context->AllocateTempInputTensor(node, kBatchToSpaceNDCropsTensor);
  TF_LITE_ENSURE(context, crops != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kBatchToSpaceNDOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) >= kBatchToSpaceNDMinDim);
  TF_LITE_ENSURE(context, NumDimensions(output) >= kBatchToSpaceNDMinDim);
  TF_LITE_ENSURE(context, NumDimensions(input) <= kBatchToSpaceNDMaxDim);
  TF_LITE_ENSURE(context, NumDimensions(output) <= kBatchToSpaceNDMaxDim);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), NumDimensions(output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, block_shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, crops->type, kTfLiteInt32);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(block_shape);
  micro_context->DeallocateTempTfLiteTensor(crops);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus BatchToSpaceNDEvalReference(TfLiteContext* context,
                                         TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kBatchToSpaceNDInputTensor);
  const TfLiteEvalTensor* block_shape = tflite::micro::GetEvalInput(
      context, node, kBatchToSpaceNDBlockShapeTensor);
  const TfLiteEvalTensor* crops =
      tflite::micro::GetEvalInput(context, node, kBatchToSpaceNDCropsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kBatchToSpaceNDOutputTensor);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      reference_ops::BatchToSpaceND(
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(block_shape),
          tflite::micro::GetTensorData<int32_t>(block_shape),
          tflite::micro::GetTensorShape(crops),
          tflite::micro::GetTensorData<int32_t>(crops),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::BatchToSpaceND(
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(block_shape),
          tflite::micro::GetTensorData<int32_t>(block_shape),
          tflite::micro::GetTensorShape(crops),
          tflite::micro::GetTensorData<int32_t>(crops),
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

}  // namespace micro
}  // namespace tflite
