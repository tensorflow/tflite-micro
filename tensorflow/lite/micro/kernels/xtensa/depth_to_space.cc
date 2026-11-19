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
#include "tensorflow/lite/micro/kernels/depth_to_space.h"

#include <cstdint>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
TfLiteStatus DepthToSpaceEvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthToSpaceInputTensor);
  if (input->type == kTfLiteInt8) {
    const auto* params =
        reinterpret_cast<const TfLiteDepthToSpaceParams*>(node->builtin_data);
    TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, kDepthToSpaceOutputTensor);
    const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

    const int num_batches = input_shape.Dims(kDepthToSpaceBatchRank);
    const int input_height = input_shape.Dims(kDepthToSpaceHeightRank);
    const int input_width = input_shape.Dims(kDepthToSpaceWidthRank);
    const int input_depth = input_shape.Dims(kDepthToSpaceDepthRank);
    const int output_height = output_shape.Dims(kDepthToSpaceHeightRank);
    const int output_width = output_shape.Dims(kDepthToSpaceWidthRank);
    const int output_depth = output_shape.Dims(kDepthToSpaceDepthRank);
    const int32_t block_size = static_cast<int32_t>(params->block_size);

    if (num_batches > 0 && input_height > 0 && input_width > 0 &&
        input_depth > 0 && output_height > 0 && output_width > 0 &&
        output_depth > 0) {
      const int input_cube_size = input_height * input_width * input_depth;
      const int output_cube_size = output_height * output_width * output_depth;
      const int8_t* p_input = tflite::micro::GetTensorData<int8_t>(input);
      int8_t* p_output = tflite::micro::GetTensorData<int8_t>(output);

      for (int b = 0; b < num_batches; ++b) {
        const int err = xa_nn_depth_to_space_8_8(
            p_output + b * output_cube_size, p_input + b * input_cube_size,
            input_height, input_width, input_depth, block_size, output_height,
            output_width, output_depth, /*inp_data_format=*/0,
            /*out_data_format=*/0);
        if (err != 0) {
          return kTfLiteError;
        }
      }
      return kTfLiteOk;
    }
  }
  return DepthToSpaceEvalReference(context, node);
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

}  // namespace

TFLMRegistration Register_DEPTH_TO_SPACE() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(nullptr, DepthToSpacePrepareReference,
                                   DepthToSpaceEvalXtensa);
#else
  return tflite::micro::RegisterOp(nullptr, DepthToSpacePrepareReference,
                                   DepthToSpaceEvalReference);
#endif
}

}  // namespace micro
}  // namespace tflite
