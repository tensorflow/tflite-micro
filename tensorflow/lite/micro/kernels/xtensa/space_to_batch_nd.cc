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
#include "tensorflow/lite/micro/kernels/space_to_batch_nd.h"

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
bool CanUseHifiSpaceToBatchND(const RuntimeShape& input_shape,
                              const int32_t* block_sizes,
                              const RuntimeShape& block_shape,
                              const int32_t* pad_sizes,
                              const RuntimeShape& paddings_shape,
                              const RuntimeShape& output_shape) {
  const int num_inp_dims = input_shape.DimensionsCount();
  const int num_out_dims = output_shape.DimensionsCount();
  if (num_inp_dims < kSpaceToBatchNDMinDim ||
      num_inp_dims > kSpaceToBatchNDMaxDim || num_out_dims != num_inp_dims) {
    return false;
  }

  const int spatial_dims = num_inp_dims - 2;
  if (block_shape.FlatSize() < spatial_dims ||
      paddings_shape.FlatSize() < 2 * spatial_dims) {
    return false;
  }

  for (int i = 0; i < num_inp_dims; ++i) {
    if (input_shape.Dims(i) <= 0 || output_shape.Dims(i) <= 0) {
      return false;
    }
  }
  int block_prod = 1;
  for (int i = 0; i < spatial_dims; ++i) {
    if (block_sizes[i] <= 0) {
      return false;
    }
    block_prod *= block_sizes[i];
  }
  for (int i = 0; i < 2 * spatial_dims; ++i) {
    if (pad_sizes[i] < 0) {
      return false;
    }
  }

  if (input_shape.Dims(0) * block_prod != output_shape.Dims(0) ||
      input_shape.Dims(num_inp_dims - 1) !=
          output_shape.Dims(num_out_dims - 1)) {
    return false;
  }
  return true;
}

TfLiteStatus SpaceToBatchNDEvalXtensa(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSpaceToBatchNDInputTensor);
  if (input->type == kTfLiteInt8) {
    const SpaceToBatchParams& params =
        *(static_cast<const SpaceToBatchParams*>(node->user_data));
    const TfLiteEvalTensor* block_shape = tflite::micro::GetEvalInput(
        context, node, kSpaceToBatchNDBlockShapeTensor);
    const TfLiteEvalTensor* crops =
        tflite::micro::GetEvalInput(context, node, kSpaceToBatchNDCropsTensor);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
        context, node, kSpaceToBatchNDOutputTensor);

    const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    const int32_t* block_sizes =
        tflite::micro::GetTensorData<int32_t>(block_shape);
    const int32_t* pad_sizes = tflite::micro::GetTensorData<int32_t>(crops);

    if (CanUseHifiSpaceToBatchND(
            input_shape, block_sizes,
            tflite::micro::GetTensorShape(block_shape), pad_sizes,
            tflite::micro::GetTensorShape(crops), output_shape)) {
      const int err = xa_nn_space_to_batch_nd_8_8(
          tflite::micro::GetTensorData<int8_t>(output), output_shape.DimsData(),
          tflite::micro::GetTensorData<int8_t>(input), input_shape.DimsData(),
          block_sizes, pad_sizes, output_shape.DimensionsCount(),
          input_shape.DimensionsCount(), params.output_offset);
      if (err != 0) {
        return kTfLiteError;
      }
      return kTfLiteOk;
    }
  }
  return SpaceToBatchNDEvalReference(context, node);
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

}  // namespace

TFLMRegistration Register_SPACE_TO_BATCH_ND() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(SpaceToBatchNDInitReference,
                                   SpaceToBatchNDPrepareReference,
                                   SpaceToBatchNDEvalXtensa);
#else
  return tflite::micro::RegisterOp(SpaceToBatchNDInitReference,
                                   SpaceToBatchNDPrepareReference,
                                   SpaceToBatchNDEvalReference);
#endif
}

}  // namespace micro
}  // namespace tflite
