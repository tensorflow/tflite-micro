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

#include "signal/src/window.h"

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
constexpr int kWeightsTensor = 1;
constexpr int kOutputTensor = 0;

struct WindowParams {
  int32_t shift = 0;
  int32_t input_size = 0;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new WindowParams;
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  params->shift = m["shift"].AsInt32();
  return params;
}

void Free(void* /*user_data*/, TfLiteOpaqueContext* /*context*/, void* data) {
  delete static_cast<WindowParams*>(data);
}

TfLiteStatus Prepare(void* /*user_data*/, TfLiteOpaqueContext* context,
                     TfLiteOpaqueNode* node) {
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfInputs(node), 2);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node), 1);

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TF_LITE_OPAQUE_ENSURE(context, input != nullptr);
  const TfLiteOpaqueTensor* weights =
      TfLiteOpaqueNodeGetInput(context, node, kWeightsTensor);
  TF_LITE_OPAQUE_ENSURE(context, weights != nullptr);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TF_LITE_OPAQUE_ENSURE(context, output != nullptr);

  TF_LITE_OPAQUE_ENSURE(context, TfLiteOpaqueTensorNumDims(input) >= 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(weights), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input),
                           TfLiteOpaqueTensorNumDims(output));

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(weights),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteInt16);

  auto* params = static_cast<WindowParams*>(TfLiteOpaqueNodeGetUserData(node));
  TF_LITE_OPAQUE_ENSURE(context, params != nullptr);
  params->input_size = GetFlatSize(input);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<WindowParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  const TfLiteOpaqueTensor* weights =
      TfLiteOpaqueNodeGetInput(context, node, kWeightsTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  const int16_t* input_data = GetTensorData<int16_t>(input);
  const int16_t* weight_data = GetTensorData<int16_t>(weights);
  int16_t* output_data = GetTensorData<int16_t>(output);
  int weight_size = TfLiteOpaqueTensorDim(weights, 0);

  for (int i = 0; i < params->input_size; i += weight_size) {
    ::tflm_signal::ApplyWindow(&input_data[i], weight_data, weight_size,
                               params->shift, &output_data[i]);
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateWindowOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r = TfLiteOperatorCreate(kTfLiteBuiltinCustom,
                                             "SignalWindow", /*version=*/1,
                                             /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(r, Init);
    TfLiteOperatorSetFreeWithData(r, Free);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_WINDOW() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalWindow", CreateWindowOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
