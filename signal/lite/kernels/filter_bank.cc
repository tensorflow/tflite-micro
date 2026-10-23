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

#include "signal/src/filter_bank.h"

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
constexpr int kWeightTensor = 1;
constexpr int kUnweightTensor = 2;
constexpr int kChFreqStartsTensor = 3;
constexpr int kChWeightStartsTensor = 4;
constexpr int kChannelWidthsTensor = 5;
constexpr int kOutputTensor = 0;

struct FilterBankParams {
  tflm_signal::FilterbankConfig config{};
  std::vector<uint64_t> work_area;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new FilterBankParams;
  const flexbuffers::Map m = GetFlexbufferMap(buffer, length);
  params->config.num_channels = m["num_channels"].AsInt32();
  params->work_area.resize(params->config.num_channels + 1);
  return params;
}

TfLiteStatus Prepare(void* /*user_data*/, TfLiteOpaqueContext* context,
                     TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfInputs(node), 6);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node), 1);

  const TfLiteOpaqueTensor* input0 =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TF_LITE_OPAQUE_ENSURE(context, input0 != nullptr);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input0), 1);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input0),
                                 kTfLiteUInt32);

  for (int i = 1; i < 6; ++i) {
    const TfLiteOpaqueTensor* input =
        TfLiteOpaqueNodeGetInput(context, node, i);
    TF_LITE_OPAQUE_ENSURE(context, input != nullptr);
    TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
    TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                   kTfLiteInt16);
  }

  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TF_LITE_OPAQUE_ENSURE(context, output != nullptr);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteUInt64);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params =
      static_cast<FilterBankParams*>(TfLiteOpaqueNodeGetUserData(node));

  const TfLiteOpaqueTensor* input0 =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  const TfLiteOpaqueTensor* input1 =
      TfLiteOpaqueNodeGetInput(context, node, kWeightTensor);
  const TfLiteOpaqueTensor* input2 =
      TfLiteOpaqueNodeGetInput(context, node, kUnweightTensor);
  const TfLiteOpaqueTensor* input3 =
      TfLiteOpaqueNodeGetInput(context, node, kChFreqStartsTensor);
  const TfLiteOpaqueTensor* input4 =
      TfLiteOpaqueNodeGetInput(context, node, kChWeightStartsTensor);
  const TfLiteOpaqueTensor* input5 =
      TfLiteOpaqueNodeGetInput(context, node, kChannelWidthsTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  params->config.weights = GetTensorData<int16_t>(input1);
  params->config.unweights = GetTensorData<int16_t>(input2);
  params->config.channel_frequency_starts = GetTensorData<int16_t>(input3);
  params->config.channel_weight_starts = GetTensorData<int16_t>(input4);
  params->config.channel_widths = GetTensorData<int16_t>(input5);

  const uint32_t* input_data = GetTensorData<uint32_t>(input0);
  uint64_t* output_data = GetTensorData<uint64_t>(output);
  tflm_signal::FilterbankAccumulateChannels(&params->config, input_data,
                                            params->work_area.data());

  // Discard channel 0, which is scratch space.
  memcpy(output_data, params->work_area.data() + 1,
         TfLiteOpaqueTensorByteSize(output));
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateFilterBankOperator() {
  static TfLiteOperator* op = MakeOperator(
      "SignalFilterBank", Init, DeleteParams<FilterBankParams>, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_FILTER_BANK() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalFilterBank", CreateFilterBankOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
