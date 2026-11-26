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
#include "tensorflow/lite/micro/kernels/slice.h"

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
bool CanUseHifiSlice(const SliceParams& op_params,
                     const RuntimeShape& extended_input_shape,
                     int32_t start[kSliceMaxDim], int32_t stop[kSliceMaxDim]) {
  for (int i = 0; i < kSliceMaxDim; ++i) {
    const int padded_i = kSliceMaxDim - i;
    start[i] = op_params.begin_count < padded_i
                   ? 0
                   : op_params.begin[op_params.begin_count - padded_i];
    stop[i] = (op_params.size_count < padded_i ||
               op_params.size[op_params.size_count - padded_i] == -1)
                  ? extended_input_shape.Dims(i)
                  : start[i] + op_params.size[op_params.size_count - padded_i];
  }

  // xa_nn_strided_slice_int* supports up to 4D slices (start_0 == 0, stop_0 ==
  // 1) with strictly positive dimensions and slice lengths.
  if (start[0] != 0 || stop[0] != 1) {
    return false;
  }
  for (int i = 1; i < kSliceMaxDim; ++i) {
    const int dim = extended_input_shape.Dims(i);
    if (dim <= 0 || start[i] < 0 || stop[i] <= start[i] || stop[i] > dim) {
      return false;
    }
  }
  return true;
}

TfLiteStatus SliceEvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSliceInputTensor);
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16 ||
      input->type == kTfLiteInt32) {
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

    const RuntimeShape extended_input_shape = RuntimeShape::ExtendedShape(
        kSliceMaxDim, tflite::micro::GetTensorShape(input));
    int32_t start[kSliceMaxDim];
    int32_t stop[kSliceMaxDim];
    if (CanUseHifiSlice(op_params, extended_input_shape, start, stop)) {
      constexpr int32_t kUnitStride = 1;
      int err = 0;
      if (input->type == kTfLiteInt8) {
        err = xa_nn_strided_slice_int8(
            tflite::micro::GetTensorData<int8_t>(output),
            tflite::micro::GetTensorData<int8_t>(input), start[0], stop[0],
            start[1], stop[1], start[2], stop[2], start[3], stop[3], start[4],
            stop[4], kUnitStride, kUnitStride, kUnitStride, kUnitStride,
            kUnitStride, extended_input_shape.Dims(1),
            extended_input_shape.Dims(2), extended_input_shape.Dims(3),
            extended_input_shape.Dims(4));
      } else if (input->type == kTfLiteInt16) {
        err = xa_nn_strided_slice_int16(
            tflite::micro::GetTensorData<int16_t>(output),
            tflite::micro::GetTensorData<int16_t>(input), start[0], stop[0],
            start[1], stop[1], start[2], stop[2], start[3], stop[3], start[4],
            stop[4], kUnitStride, kUnitStride, kUnitStride, kUnitStride,
            kUnitStride, extended_input_shape.Dims(1),
            extended_input_shape.Dims(2), extended_input_shape.Dims(3),
            extended_input_shape.Dims(4));
      } else {
        err = xa_nn_strided_slice_int32(
            tflite::micro::GetTensorData<int32_t>(output),
            tflite::micro::GetTensorData<int32_t>(input), start[0], stop[0],
            start[1], stop[1], start[2], stop[2], start[3], stop[3], start[4],
            stop[4], kUnitStride, kUnitStride, kUnitStride, kUnitStride,
            kUnitStride, extended_input_shape.Dims(1),
            extended_input_shape.Dims(2), extended_input_shape.Dims(3),
            extended_input_shape.Dims(4));
      }
      if (err != 0) {
        return kTfLiteError;
      }
      return kTfLiteOk;
    }
  }
  return SliceEvalReference(context, node);
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

}  // namespace

TFLMRegistration Register_SLICE() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(nullptr, SlicePrepareReference,
                                   SliceEvalXtensa);
#else
  return tflite::micro::RegisterOp(nullptr, SlicePrepareReference,
                                   SliceEvalReference);
#endif
}

}  // namespace micro
}  // namespace tflite
