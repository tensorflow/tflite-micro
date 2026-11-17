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
#include "tensorflow/lite/micro/kernels/batch_to_space_nd.h"

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
bool CanUseHifiBatchToSpaceND(const RuntimeShape& input_shape,
                              const int32_t* block_sizes,
                              const RuntimeShape& block_shape,
                              const int32_t* crop_sizes,
                              const RuntimeShape& crops_shape,
                              const RuntimeShape& output_shape,
                              int32_t out_shape_4d[4]) {
  const int num_inp_dims = input_shape.DimensionsCount();
  const int num_out_dims = output_shape.DimensionsCount();
  if (num_inp_dims < kBatchToSpaceNDMinDim ||
      num_inp_dims > kBatchToSpaceNDMaxDim || num_out_dims != num_inp_dims) {
    return false;
  }

  const int spatial_dims = num_inp_dims - 2;
  if (block_shape.FlatSize() < spatial_dims ||
      crops_shape.FlatSize() < 2 * spatial_dims) {
    return false;
  }

  if (num_out_dims == 3) {
    out_shape_4d[0] = output_shape.Dims(0);
    out_shape_4d[1] = output_shape.Dims(1);
    out_shape_4d[2] = 1;
    out_shape_4d[3] = output_shape.Dims(2);
  } else {
    for (int i = 0; i < 4; ++i) {
      out_shape_4d[i] = output_shape.Dims(i);
    }
  }

  for (int i = 0; i < num_inp_dims; ++i) {
    if (input_shape.Dims(i) <= 0) {
      return false;
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (out_shape_4d[i] <= 0) {
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
    if (crop_sizes[i] < 0) {
      return false;
    }
  }

  if (input_shape.Dims(0) != out_shape_4d[0] * block_prod ||
      input_shape.Dims(num_inp_dims - 1) != out_shape_4d[3]) {
    return false;
  }
  return true;
}

TfLiteStatus BatchToSpaceNDEvalXtensa(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kBatchToSpaceNDInputTensor);
  if (input->type == kTfLiteInt8) {
    const TfLiteEvalTensor* block_shape = tflite::micro::GetEvalInput(
        context, node, kBatchToSpaceNDBlockShapeTensor);
    const TfLiteEvalTensor* crops =
        tflite::micro::GetEvalInput(context, node, kBatchToSpaceNDCropsTensor);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
        context, node, kBatchToSpaceNDOutputTensor);

    const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    const int32_t* block_sizes =
        tflite::micro::GetTensorData<int32_t>(block_shape);
    const int32_t* crop_sizes = tflite::micro::GetTensorData<int32_t>(crops);
    int32_t out_shape_4d[4];

    if (CanUseHifiBatchToSpaceND(
            input_shape, block_sizes,
            tflite::micro::GetTensorShape(block_shape), crop_sizes,
            tflite::micro::GetTensorShape(crops), output_shape, out_shape_4d)) {
      const int err = xa_nn_batch_to_space_nd_8_8(
          tflite::micro::GetTensorData<int8_t>(output), out_shape_4d,
          tflite::micro::GetTensorData<int8_t>(input), input_shape.DimsData(),
          block_sizes, crop_sizes, /*num_out_dims=*/4,
          input_shape.DimensionsCount());
      if (err != 0) {
        return kTfLiteError;
      }
      return kTfLiteOk;
    }
  }
  return BatchToSpaceNDEvalReference(context, node);
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

}  // namespace

TFLMRegistration Register_BATCH_TO_SPACE_ND() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(nullptr, BatchToSpaceNDPrepareReference,
                                   BatchToSpaceNDEvalXtensa);
#else
  return tflite::micro::RegisterOp(nullptr, BatchToSpaceNDPrepareReference,
                                   BatchToSpaceNDEvalReference);
#endif
}

}  // namespace micro
}  // namespace tflite
