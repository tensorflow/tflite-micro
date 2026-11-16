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

#include "tensorflow/lite/micro/kernels/activations.h"

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const ReluOpData& data = *(static_cast<const ReluOpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
      const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
      int err = xa_nn_vec_relu_std_f32_f32(
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorData<float>(input), flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      ReluEvalReference<float>(data, input, output);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
      int err = xa_nn_vec_relu_asym8s_asym8s(
          tflite::micro::GetTensorData<int8_t>(output),
          tflite::micro::GetTensorData<int8_t>(input), data.params.input_offset,
          data.params.output_multiplier, data.params.output_shift,
          data.params.output_offset, data.params.quantized_activation_min,
          data.params.quantized_activation_max, flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      ReluEvalReference<int8_t>(data, input, output);
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      ReluEvalReference<int16_t>(data, input, output);
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32/int8/int16 is supported currently, got %s",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const Relu6OpData& data = *(static_cast<const Relu6OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
      const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
      int err = xa_nn_vec_relu6_f32_f32(
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorData<float>(input), flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      Relu6EvalReference<float>(data, input, output);
#endif  // defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5))
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
      int err = xa_nn_vec_activation_min_max_8_8(
          tflite::micro::GetTensorData<int8_t>(output),
          tflite::micro::GetTensorData<int8_t>(input), data.zero, data.six,
          flat_size);
      if (err != 0) {
        return kTfLiteError;
      }
#else
      Relu6EvalReference<int8_t>(data, input, output);
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      Relu6EvalReference<int16_t>(data, input, output);
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32/int8/int16 is supported currently, got %s",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_RELU() {
  return tflite::micro::RegisterOp(ReluInit, ReluPrepare, ReluEval);
}

TFLMRegistration Register_RELU6() {
  return tflite::micro::RegisterOp(Relu6Init, Relu6Prepare, Relu6Eval);
}

}  // namespace micro
}  // namespace tflite
