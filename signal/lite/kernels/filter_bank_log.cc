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

#include "signal/src/filter_bank_log.h"

#include <stddef.h>
#include <stdint.h>

#include "flatbuffers/flexbuffers.h"
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

struct FilterBankLogParams {
  int32_t output_scale = 0;
  int32_t input_correction_bits = 0;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new FilterBankLogParams;
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  params->output_scale = m["output_scale"].AsInt32();
  params->input_correction_bits = m["input_correction_bits"].AsInt32();
  return params;
}

void Free(void* /*user_data*/, TfLiteOpaqueContext* /*context*/, void* data) {
  delete static_cast<FilterBankLogParams*>(data);
}

TfLiteStatus Prepare(void* /*user_data*/, TfLiteOpaqueContext* context,
                     TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfInputs(node), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node), 1);

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TF_LITE_OPAQUE_ENSURE(context, input != nullptr);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TF_LITE_OPAQUE_ENSURE(context, output != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteUInt32);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteInt16);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params =
      static_cast<FilterBankLogParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  const uint32_t* input_data = GetTensorData<uint32_t>(input);
  int16_t* output_data = GetTensorData<int16_t>(output);
  int num_channels = TfLiteOpaqueTensorDim(input, 0);

  tflm_signal::FilterbankLog(input_data, num_channels, params->output_scale,
                             params->input_correction_bits, output_data);
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateFilterBankLogOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r = TfLiteOperatorCreate(
        kTfLiteBuiltinCustom, "SignalFilterBankLog", /*version=*/1,
        /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(r, Init);
    TfLiteOperatorSetFreeWithData(r, Free);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_FILTER_BANK_LOG() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalFilterBankLog", CreateFilterBankLogOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
