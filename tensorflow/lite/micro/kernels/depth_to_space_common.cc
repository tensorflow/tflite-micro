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
#include "tensorflow/lite/kernels/internal/reference/depth_to_space.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/depth_to_space.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

const int kDepthToSpaceInputTensor = 0;
const int kDepthToSpaceOutputTensor = 0;

// input/output tensor shape rank associations
const int kBatchRank = 0;
const int kHeightRank = 1;
const int kWidthRank = 2;
const int kDepthRank = 3;

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthToSpaceParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDepthToSpaceInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kDepthToSpaceOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);

  auto data_type = output->type;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  const int block_size = params->block_size;
  TF_LITE_ENSURE(context, block_size > 0);
  const int input_height = input->dims->data[kHeightRank];
  const int input_width = input->dims->data[kWidthRank];
  const int input_channels = input->dims->data[kDepthRank];
  int output_height = input_height * block_size;
  int output_width = input_width * block_size;
  int output_channels = input_channels / block_size / block_size;

  TF_LITE_ENSURE_EQ(context, input_height, output_height / block_size);
  TF_LITE_ENSURE_EQ(context, input_width, output_width / block_size);
  TF_LITE_ENSURE_EQ(context, input_channels,
                    output_channels * block_size * block_size);

  // We must update the output tensor dimensions.
  // The dims storage is expected to be the same area in memory
  // for both TfLiteTensor and TfLiteEvalTensor.  This is important
  // because TfLiteTensor in the MicroInterpreter is a temporary
  // allocation.  For the KernelRunner interpreter, TfLiteEvalTensor
  // is a temporary allocation.  We must therefore relocate the dims
  // from the FlatBuffer to the persistant storage arena.
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kDepthToSpaceOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));
  output->dims->data[kBatchRank] = input->dims->data[kBatchRank];
  output->dims->data[kHeightRank] = output_height;
  output->dims->data[kWidthRank] = output_width;
  output->dims->data[kDepthRank] = output_channels;

  return kTfLiteOk;
}

TfLiteStatus DepthToSpacePrepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

}  // namespace tflite
