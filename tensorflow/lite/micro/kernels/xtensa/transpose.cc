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
#include "tensorflow/lite/micro/kernels/transpose.h"

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
bool CanUseHifiTranspose(const RuntimeShape& input_shape,
                         const RuntimeShape& output_shape,
                         const int32_t* perm_data, int size) {
  const int num_inp_dims = input_shape.DimensionsCount();
  const int num_out_dims = output_shape.DimensionsCount();
  if (num_inp_dims <= 0 || num_inp_dims > 5 || num_out_dims != num_inp_dims ||
      size != num_inp_dims) {
    return false;
  }
  for (int i = 0; i < num_inp_dims; ++i) {
    if (input_shape.Dims(i) <= 0 || output_shape.Dims(i) <= 0 ||
        perm_data[i] < 0 || perm_data[i] >= num_inp_dims ||
        output_shape.Dims(i) != input_shape.Dims(perm_data[i])) {
      return false;
    }
  }
  return true;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

TfLiteStatus TransposeEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTransposeInputTensor);
  if (input->type != kTfLiteInt8) {
    MicroPrintf("Type %s (%d) is not supported by Transpose_INT8.",
                TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const TfLiteEvalTensor* perm_tensor =
      tflite::micro::GetEvalInput(context, node, kTransposePermTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTransposeOutputTensor);
  const int32_t* perm_data = tflite::micro::GetTensorData<int32_t>(perm_tensor);
  const int size = perm_tensor->dims->data[0];
  const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  if (CanUseHifiTranspose(input_shape, output_shape, perm_data, size)) {
    const int err = xa_nn_transpose_8_8(
        tflite::micro::GetTensorData<int8_t>(output), output_shape.DimsData(),
        tflite::micro::GetTensorData<int8_t>(input), input_shape.DimsData(),
        perm_data, output_shape.DimensionsCount(),
        input_shape.DimensionsCount());
    if (err != 0) {
      return kTfLiteError;
    }
    return kTfLiteOk;
  }
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

  return TransposeEvalReference(context, node);
}

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
TfLiteStatus TransposeEvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTransposeInputTensor);
  if (input->type == kTfLiteInt8) {
    return TransposeEvalInt8(context, node);
  }
  if (input->type == kTfLiteInt16) {
    const TfLiteEvalTensor* perm_tensor =
        tflite::micro::GetEvalInput(context, node, kTransposePermTensor);
    TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, kTransposeOutputTensor);
    const int32_t* perm_data =
        tflite::micro::GetTensorData<int32_t>(perm_tensor);
    const int size = perm_tensor->dims->data[0];
    const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

    if (CanUseHifiTranspose(input_shape, output_shape, perm_data, size)) {
      const int err = xa_nn_transpose_16_16(
          tflite::micro::GetTensorData<int16_t>(output),
          output_shape.DimsData(), tflite::micro::GetTensorData<int16_t>(input),
          input_shape.DimsData(), perm_data, output_shape.DimensionsCount(),
          input_shape.DimensionsCount());
      if (err != 0) {
        return kTfLiteError;
      }
      return kTfLiteOk;
    }
  }
  return TransposeEvalReference(context, node);
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

}  // namespace

TFLMRegistration Register_TRANSPOSE() {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(nullptr, TransposePrepare,
                                   TransposeEvalXtensa);
#else
  return tflite::micro::RegisterOp(nullptr, TransposePrepare,
                                   TransposeEvalReference);
#endif
}

TFLMRegistration Register_TRANSPOSE_INT8() {
  return tflite::micro::RegisterOp(nullptr, TransposePrepare,
                                   TransposeEvalInt8);
}

}  // namespace micro
}  // namespace tflite
