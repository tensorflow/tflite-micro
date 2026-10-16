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

#include "signal/src/fft_auto_scale.h"

#include <stddef.h>
#include <stdint.h>

#include "signal/lite/kernels/kernel_util.h"
#include "signal/lite/kernels/signal_ops.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kScaleBitTensor = 1;

TfLiteStatus Prepare(void* /*user_data*/, TfLiteOpaqueContext* context,
                     TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfInputs(node), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node), 2);

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TF_LITE_OPAQUE_ENSURE(context, input != nullptr);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TF_LITE_OPAQUE_ENSURE(context, output != nullptr);
  TfLiteOpaqueTensor* scale_bit =
      TfLiteOpaqueNodeGetOutput(context, node, kScaleBitTensor);
  TF_LITE_OPAQUE_ENSURE(context, scale_bit != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(scale_bit), 0);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(scale_bit),
                                 kTfLiteInt32);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TfLiteOpaqueTensor* scale_bit =
      TfLiteOpaqueNodeGetOutput(context, node, kScaleBitTensor);

  const int16_t* input_data = GetTensorData<int16_t>(input);
  int16_t* output_data = GetTensorData<int16_t>(output);
  int32_t* scale_bit_data = GetTensorData<int32_t>(scale_bit);

  *scale_bit_data = tflm_signal::FftAutoScale(
      input_data, TfLiteOpaqueTensorDim(output, 0), output_data);
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateFftAutoScaleOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r = TfLiteOperatorCreate(
        kTfLiteBuiltinCustom, "SignalFftAutoScale", /*version=*/1,
        /*user_data=*/nullptr);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_FFT_AUTO_SCALE() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalFftAutoScale", CreateFftAutoScaleOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
