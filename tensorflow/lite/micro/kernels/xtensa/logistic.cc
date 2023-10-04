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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/logistic.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* LogisticInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataLogistic));
}

TfLiteStatus LogisticEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kLogisticInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kLogisticOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataLogistic* data = static_cast<OpDataLogistic*>(node->user_data);

  if (input->type != output->type) {
    MicroPrintf(
        "Input and output types must be identical. Input %s, output %s.",
        TfLiteTypeGetName(input->type), TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteFloat32: {
#if HIFI_VFPU && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      const float* inp_data_ptr = tflite::micro::GetTensorData<float>(input);
      float* out_data_ptr = tflite::micro::GetTensorData<float>(output);

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_vec_sigmoid_f32_f32(out_data_ptr, inp_data_ptr, flat_size), 0);
#else
      reference_ops::Logistic(tflite::micro::GetTensorShape(input),
                              tflite::micro::GetTensorData<float>(input),
                              tflite::micro::GetTensorShape(output),
                              tflite::micro::GetTensorData<float>(output));
#endif  // HIFI_VFPU && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
      break;
    }
    case kTfLiteInt8: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      const int8_t* input_data_ptr =
          tflite::micro::GetTensorData<int8_t>(input);
      int8_t* output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_vec_sigmoid_asym8s_asym8s(
              output_data_ptr, input_data_ptr, data->input_zero_point,
              data->input_range_radius, data->input_multiplier,
              data->input_left_shift, flat_size),
          0);
#else
      reference_integer_ops::Logistic(
          data->input_zero_point, data->input_range_radius,
          data->input_multiplier, data->input_left_shift,
          NumElements(input->dims), tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output));
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      break;
    }
    case kTfLiteInt16: {
      switch (output->type) {
        case kTfLiteInt16:
          reference_integer_ops::Logistic(
              data->input_multiplier, data->input_left_shift,
              NumElements(input->dims),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        default:
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
      break;
    }
    default: {
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_LOGISTIC() {
  return tflite::micro::RegisterOp(LogisticInit, LogisticPrepare, LogisticEval);
}
}  // namespace tflite
