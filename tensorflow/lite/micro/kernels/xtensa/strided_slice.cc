/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"

#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
void StridedSlice_int16_hifi4opt(const tflite::StridedSliceParams& op_params,
                                 const RuntimeShape& unextended_input_shape,
                                 const int16_t* input_data,
                                 const RuntimeShape& unextended_output_shape,
                                 int16_t* output_data) {
  using ::tflite::strided_slice::StartForAxis;
  using ::tflite::strided_slice::StopForAxis;

  ruy::profiler::ScopeLabel label("StridedSlice");

  // Note that the output_shape is not used herein.
  tflite::StridedSliceParams params_copy = op_params;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 5);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(5, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(5, unextended_output_shape);

  // Reverse and pad to 5 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 5D and are given backwards).
  ::tflite::strided_slice::StridedSlicePadIndices(&params_copy, 5);

  const int start_0 = StartForAxis(params_copy, input_shape, 0);
  const int stop_0 = StopForAxis(params_copy, input_shape, 0, start_0);
  const int start_1 = StartForAxis(params_copy, input_shape, 1);
  const int stop_1 = StopForAxis(params_copy, input_shape, 1, start_1);
  const int start_2 = StartForAxis(params_copy, input_shape, 2);
  const int stop_2 = StopForAxis(params_copy, input_shape, 2, start_2);
  const int start_3 = StartForAxis(params_copy, input_shape, 3);
  const int stop_3 = StopForAxis(params_copy, input_shape, 3, start_3);
  const int start_4 = StartForAxis(params_copy, input_shape, 4);
  const int stop_4 = StopForAxis(params_copy, input_shape, 4, start_4);

  xa_nn_strided_slice_int16(output_data, input_data, static_cast<int>(start_0),
                            static_cast<int>(stop_0), static_cast<int>(start_1),
                            static_cast<int>(stop_1), static_cast<int>(start_2),
                            static_cast<int>(stop_2), static_cast<int>(start_3),
                            static_cast<int>(stop_3), static_cast<int>(start_4),
                            static_cast<int>(stop_4), params_copy.strides[0],
                            params_copy.strides[1], params_copy.strides[2],
                            params_copy.strides[3], params_copy.strides[4],
                            input_shape.Dims(1), input_shape.Dims(2),
                            input_shape.Dims(3), input_shape.Dims(4));
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const StridedSliceParams& op_params =
      *(static_cast<const StridedSliceParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kStridedSliceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kStridedSliceOutputTensor);
  switch (output->type) {
    case kTfLiteFloat32:
      reference_ops::StridedSlice(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<float>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::StridedSlice(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<int8_t>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      StridedSlice_int16_hifi4opt(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
#else
      reference_ops::StridedSlice(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      break;
    case kTfLiteInt32:
      reference_ops::StridedSlice(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int32_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteBool:
      reference_ops::StridedSlice(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<bool>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<bool>(output));
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TFLMRegistration Register_STRIDED_SLICE() {
  return tflite::micro::RegisterOp(StridedSliceInit, StridedSlicePrepare, Eval);
}

}  // namespace tflite
