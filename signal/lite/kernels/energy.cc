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

#include "signal/src/energy.h"

#include <stddef.h>
#include <stdint.h>

#include "flatbuffers/flexbuffers.h"
#include "signal/lite/kernels/kernel_util.h"
#include "signal/lite/kernels/signal_ops.h"
#include "signal/src/complex.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct EnergyParams {
  int32_t start_index = 0;
  int32_t end_index = 0;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new EnergyParams;
  const flexbuffers::Map m = GetFlexbufferMap(buffer, length);
  params->start_index = m["start_index"].AsInt32();
  params->end_index = m["end_index"].AsInt32();
  return params;
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
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteUInt32);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<EnergyParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  const Complex<int16_t>* input_data = GetTensorData<Complex<int16_t>>(input);
  uint32_t* output_data = GetTensorData<uint32_t>(output);

  tflm_signal::SpectrumToEnergy(input_data, params->start_index,
                                params->end_index,
                                TfLiteOpaqueTensorDim(output, 0), output_data);
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateEnergyOperator() {
  static TfLiteOperator* op = MakeOperator(
      "SignalEnergy", Init, DeleteParams<EnergyParams>, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_ENERGY() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalEnergy", CreateEnergyOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
