/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  int32_t input_zero_point;
  int32_t input_range_radius;
  int32_t input_multiplier;
  int input_left_shift;
};

void* TanhInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus CalculateArithmeticOpData(TfLiteContext* context, TfLiteNode* node,
                                       OpData* data) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    static constexpr int kInputIntegerBits = 4;
    const double input_real_multiplier =
        static_cast<double>(input->params.scale) *
        static_cast<double>(1 << (31 - kInputIntegerBits));

    const double q = std::frexp(input_real_multiplier, &data->input_left_shift);
    data->input_multiplier = static_cast<int32_t>(TfLiteRound(q * (1ll << 31)));

    data->input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 31);
  }
  return kTfLiteOk;
}

TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  data->input_zero_point = input->params.zero_point;
  return CalculateArithmeticOpData(context, node, data);
}

}  // namespace

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Tanh(tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<float>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      TanhParams params;
      params.input_left_shift = data.input_left_shift;
      reference_ops::Tanh(params, tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<int16_t>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      TanhParams params;
      params.input_zero_point = data.input_zero_point;
      params.input_range_radius = data.input_range_radius;
      params.input_multiplier = data.input_multiplier;
      params.input_left_shift = data.input_left_shift;
      reference_ops::Tanh(params, tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<uint8_t>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<uint8_t>(output));

      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
#if defined(HIFI5)
      int err;
      const int8_t *input_data_ptr;
      int8_t *output_data_ptr;
      const RuntimeShape& input_shape  = tflite::micro::GetTensorShape(input);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int flat_size = MatchingFlatSize(input_shape, output_shape);

      input_data_ptr  = tflite::micro::GetTensorData<int8_t>(input);
      output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

      err = xa_nn_vec_tanh_asym8s_asym8s(output_data_ptr,
                                        input_data_ptr,
                                        data.input_zero_point,
                                        data.input_range_radius,
                                        data.input_multiplier,
                                        data.input_left_shift,
                                        flat_size);
      TF_LITE_MICRO_EXPECT_EQ(0, err);
#else
      reference_integer_ops::Tanh(
          data.input_zero_point, data.input_range_radius, data.input_multiplier,
          data.input_left_shift, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif // defined(HIFI5)
      return kTfLiteOk;
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                         TfLiteTypeGetName(input->type),
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration Register_TANH() {
  return {/*init=*/activations::TanhInit,
          /*free=*/nullptr,
          /*prepare=*/activations::TanhPrepare,
          /*invoke=*/activations::TanhEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite
