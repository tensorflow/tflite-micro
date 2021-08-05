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

TfLiteStatus DepthToSpaceEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthToSpaceParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthToSpaceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthToSpaceOutputTensor);

  tflite::DepthToSpaceParams op_params;
  op_params.block_size = static_cast<int32_t>(params->block_size);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      reference_ops::DepthToSpace(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<float>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::DepthToSpace(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<int8_t>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<int8_t>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "DEPTH_TO_SPACE only supports FLOAT32 and INT8, got %s.",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteRegistration Register_DEPTH_TO_SPACE() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/DepthToSpacePrepare,
          /*invoke=*/DepthToSpaceEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
