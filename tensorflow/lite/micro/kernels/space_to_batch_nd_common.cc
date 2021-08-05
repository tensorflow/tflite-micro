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

#include "tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/space_to_batch_nd.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

const int kSpaceToBatchNDInputTensor = 0;
const int kSpaceToBatchNDBlockShapeTensor = 1;
const int kSpaceToBatchNDCropsTensor = 2;
const int kSpaceToBatchNDOutputTensor = 0;

// Currently, only 3D NHC and 4D NHWC input/output op_context are supported.
// In case of 3D input, it will be extended to 3D NHWC by adding W=1.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(b/149952582): Support arbitrary dimension in SpaceToBatchND.
const int kInputOutputMinDimensionNum = 3;
const int kInputOutputMaxDimensionNum = 4;

TfLiteStatus SpaceToBatchNDPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kSpaceToBatchNDInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kSpaceToBatchNDOutputTensor);
  TF_LITE_ENSURE(context, input != nullptr && output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) >= kInputOutputMinDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(output) >= kInputOutputMinDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(input) <= kInputOutputMaxDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(output) <= kInputOutputMaxDimensionNum);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  return kTfLiteOk;
}

}  // namespace tflite
