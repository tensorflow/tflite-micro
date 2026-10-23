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

#include "signal/src/filter_bank_square_root.h"

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
constexpr int kScaleBitsTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(void* /*user_data*/, TfLiteOpaqueContext* context,
                     TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfInputs(node), 2);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node), 1);

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TF_LITE_OPAQUE_ENSURE(context, input != nullptr);
  const TfLiteOpaqueTensor* scale_bits =
      TfLiteOpaqueNodeGetInput(context, node, kScaleBitsTensor);
  TF_LITE_OPAQUE_ENSURE(context, scale_bits != nullptr);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TF_LITE_OPAQUE_ENSURE(context, output != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(scale_bits), 0);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteUInt64);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(scale_bits),
                                 kTfLiteInt32);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteUInt32);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  const TfLiteOpaqueTensor* scale_bits =
      TfLiteOpaqueNodeGetInput(context, node, kScaleBitsTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  const uint64_t* input_data = GetTensorData<uint64_t>(input);
  const int32_t* scale_bits_data = GetTensorData<int32_t>(scale_bits);
  uint32_t* output_data = GetTensorData<uint32_t>(output);
  int32_t num_channels = TfLiteOpaqueTensorDim(input, 0);

  tflm_signal::FilterbankSqrt(input_data, num_channels, *scale_bits_data,
                              output_data);
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateFilterBankSquareRootOperator() {
  static TfLiteOperator* op = MakeOperator("SignalFilterBankSquareRoot",
                                           nullptr, nullptr, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_FILTER_BANK_SQUARE_ROOT() {
  static ::TfLiteRegistration r = MakeRegistration(
      "SignalFilterBankSquareRoot", CreateFilterBankSquareRootOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
