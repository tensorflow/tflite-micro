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

#include "tensorflow/lite/micro/kernels/internal/reference/slice.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/slice.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {

TfLiteStatus PopulateSliceParams(const TfLiteEvalTensor* input,
                                 const TfLiteEvalTensor* begin,
                                 const TfLiteEvalTensor* size,
                                 SliceParams* op_params) {
  op_params->begin_count = kSliceMaxDim;
  op_params->size_count = kSliceMaxDim;
  for (int i = 0; i < kSliceMaxDim; ++i) {
    op_params->begin[i] = 0;
    op_params->size[i] = 1;
  }

  if (begin->type == kTfLiteInt32) {
    GetSliceBeginAndSizeVectors<int32_t>(input->dims->size, begin, size,
                                         op_params->begin, op_params->size);
  } else if (begin->type == kTfLiteInt64) {
    GetSliceBeginAndSizeVectors<int64_t>(input->dims->size, begin, size,
                                         op_params->begin, op_params->size);
  } else {
    MicroPrintf("Begin tensor type %s (%d) not supported.",
                TfLiteTypeGetName(begin->type), begin->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SlicePrepareReference(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kSliceInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* begin =
      micro_context->AllocateTempInputTensor(node, kSliceBeginTensor);
  TF_LITE_ENSURE(context, begin != nullptr);
  TfLiteTensor* size =
      micro_context->AllocateTempInputTensor(node, kSliceSizeTensor);
  TF_LITE_ENSURE(context, size != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kSliceOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Ensure validity of input tensor and its dimension.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, begin->type, size->type);
  TF_LITE_ENSURE(context,
                 begin->type == kTfLiteInt32 || begin->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, NumDimensions(begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(begin), NumElements(size));
  TF_LITE_ENSURE_EQ(context, NumElements(begin), NumDimensions(input));
  TF_LITE_ENSURE(context, NumDimensions(input) <= kSliceMaxDim);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(begin);
  micro_context->DeallocateTempTfLiteTensor(size);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus SliceEvalReference(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSliceInputTensor);
  const TfLiteEvalTensor* begin =
      tflite::micro::GetEvalInput(context, node, kSliceBeginTensor);
  const TfLiteEvalTensor* size =
      tflite::micro::GetEvalInput(context, node, kSliceSizeTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSliceOutputTensor);

  SliceParams op_params;
  TfLiteStatus status = PopulateSliceParams(input, begin, size, &op_params);
  if (status != kTfLiteOk) {
    return status;
  }

  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Slice<float>(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<float>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt32:
      reference_ops::Slice<int32_t>(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int32_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteInt8:
      reference_ops::Slice<int8_t>(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::Slice<int16_t>(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
      break;
    case kTfLiteBool:
      reference_ops::Slice<bool>(op_params,
                                 tflite::micro::GetTensorShape(input),
                                 tflite::micro::GetTensorData<bool>(input),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<bool>(output));
      break;
    default:
      MicroPrintf("Input tensor type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
