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

#include "signal/src/filter_bank_spectral_subtraction.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <vector>

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
constexpr int kNoiseEstimateTensor = 1;

struct FilterBankSpectralSubtractionParams {
  tflm_signal::SpectralSubtractionConfig config{};
  std::vector<uint32_t> noise_estimate;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new FilterBankSpectralSubtractionParams;
  const flexbuffers::Map m = GetFlexbufferMap(buffer, length);
  params->config.smoothing = m["smoothing"].AsInt32();
  params->config.one_minus_smoothing = m["one_minus_smoothing"].AsInt32();
  params->config.alternate_smoothing = m["alternate_smoothing"].AsInt32();
  params->config.alternate_one_minus_smoothing =
      m["alternate_one_minus_smoothing"].AsInt32();
  params->config.smoothing_bits = m["smoothing_bits"].AsInt32();
  params->config.min_signal_remaining = m["min_signal_remaining"].AsInt32();
  params->config.clamping = m["clamping"].AsBool();
  params->config.num_channels = m["num_channels"].AsInt32();
  params->config.spectral_subtraction_bits =
      m["spectral_subtraction_bits"].AsInt32();

  params->noise_estimate.assign(params->config.num_channels, 0);
  return params;
}

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
  TfLiteOpaqueTensor* noise_estimate =
      TfLiteOpaqueNodeGetOutput(context, node, kNoiseEstimateTensor);
  TF_LITE_OPAQUE_ENSURE(context, noise_estimate != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(noise_estimate),
                           1);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteUInt32);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteUInt32);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(
      context, TfLiteOpaqueTensorType(noise_estimate), kTfLiteUInt32);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<FilterBankSpectralSubtractionParams*>(
      TfLiteOpaqueNodeGetUserData(node));

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TfLiteOpaqueTensor* noise_estimate =
      TfLiteOpaqueNodeGetOutput(context, node, kNoiseEstimateTensor);

  const uint32_t* input_data = GetTensorData<uint32_t>(input);
  uint32_t* output_data = GetTensorData<uint32_t>(output);
  uint32_t* noise_estimate_data = GetTensorData<uint32_t>(noise_estimate);

  tflm_signal::FilterbankSpectralSubtraction(
      &params->config, input_data, output_data, params->noise_estimate.data());

  memcpy(noise_estimate_data, params->noise_estimate.data(),
         TfLiteOpaqueTensorByteSize(noise_estimate));
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateFilterBankSpectralSubtractionOperator() {
  static TfLiteOperator* op = MakeOperator(
      "SignalFilterBankSpectralSubtraction", Init,
      DeleteParams<FilterBankSpectralSubtractionParams>, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_FILTER_BANK_SPECTRAL_SUBTRACTION() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalFilterBankSpectralSubtraction",
                       CreateFilterBankSpectralSubtractionOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
