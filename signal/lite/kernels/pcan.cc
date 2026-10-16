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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "flatbuffers/flexbuffers.h"
#include "signal/lite/kernels/kernel_util.h"
#include "signal/lite/kernels/signal_ops.h"
#include "signal/src/pcan_argc_fixed.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kNoiseEstimateTensor = 1;
constexpr int kGainLutTensor = 2;
constexpr int kOutputTensor = 0;

struct PcanParams {
  int32_t snr_shift = 0;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new PcanParams;
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  params->snr_shift = m["snr_shift"].AsInt32();
  return params;
}

void Free(void* /*user_data*/, TfLiteOpaqueContext* /*context*/, void* data) {
  delete static_cast<PcanParams*>(data);
}

TfLiteStatus Prepare(void* /*user_data*/, TfLiteOpaqueContext* context,
                     TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfInputs(node), 3);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node), 1);

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TF_LITE_OPAQUE_ENSURE(context, input != nullptr);
  const TfLiteOpaqueTensor* noise_estimate =
      TfLiteOpaqueNodeGetInput(context, node, kNoiseEstimateTensor);
  TF_LITE_OPAQUE_ENSURE(context, noise_estimate != nullptr);
  const TfLiteOpaqueTensor* gain_lut =
      TfLiteOpaqueNodeGetInput(context, node, kGainLutTensor);
  TF_LITE_OPAQUE_ENSURE(context, gain_lut != nullptr);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TF_LITE_OPAQUE_ENSURE(context, output != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(noise_estimate),
                           1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(gain_lut), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteUInt32);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(
      context, TfLiteOpaqueTensorType(noise_estimate), kTfLiteUInt32);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(gain_lut),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteUInt32);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<PcanParams*>(TfLiteOpaqueNodeGetUserData(node));

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  const TfLiteOpaqueTensor* noise_estimate =
      TfLiteOpaqueNodeGetInput(context, node, kNoiseEstimateTensor);
  const TfLiteOpaqueTensor* gain_lut =
      TfLiteOpaqueNodeGetInput(context, node, kGainLutTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  const uint32_t* input_data = GetTensorData<uint32_t>(input);
  const uint32_t* noise_estimate_data = GetTensorData<uint32_t>(noise_estimate);
  const int16_t* gain_lut_data = GetTensorData<int16_t>(gain_lut);
  uint32_t* output_data = GetTensorData<uint32_t>(output);

  int num_channels = TfLiteOpaqueTensorDim(input, 0);
  memcpy(output_data, input_data, TfLiteOpaqueTensorByteSize(output));

  tflm_signal::ApplyPcanAutoGainControlFixed(gain_lut_data, params->snr_shift,
                                             noise_estimate_data, output_data,
                                             num_channels);
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreatePcanOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r =
        TfLiteOperatorCreate(kTfLiteBuiltinCustom, "SignalPCAN", /*version=*/1,
                             /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(r, Init);
    TfLiteOperatorSetFreeWithData(r, Free);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_PCAN() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalPCAN", CreatePcanOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
