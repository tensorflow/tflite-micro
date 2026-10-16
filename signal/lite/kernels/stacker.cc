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

#include "flatbuffers/flexbuffers.h"
#include "signal/lite/kernels/kernel_util.h"
#include "signal/lite/kernels/signal_ops.h"
#include "signal/src/circular_buffer.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kOutputValidTensor = 1;

struct StackerParams {
  int32_t num_channels = 0;
  int32_t stacker_left_context = 0;
  int32_t stacker_right_context = 0;
  int32_t stacker_step = 0;

  size_t buffer_size = 0;
  size_t step_size = 0;
  bool stacker_has_first_frame = false;

  int8_t* state = nullptr;
  tflm_signal::CircularBuffer* circular_buffer = nullptr;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new StackerParams;
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  params->num_channels = m["num_channels"].AsInt32();
  params->stacker_left_context = m["stacker_left_context"].AsInt32();
  params->stacker_right_context = m["stacker_right_context"].AsInt32();
  params->stacker_step = m["stacker_step"].AsInt32();

  params->buffer_size =
      static_cast<size_t>(params->num_channels) *
      (params->stacker_left_context + params->stacker_right_context + 1);
  params->step_size =
      static_cast<size_t>(params->num_channels) * params->stacker_step;
  params->stacker_has_first_frame = false;

  size_t state_size =
      tflm_signal::CircularBufferGetNeededMemory(params->buffer_size);
  params->state = new int8_t[state_size];
  params->circular_buffer = tflm_signal::CircularBufferInit(
      params->buffer_size, params->state, state_size);
  return params;
}

void Free(void* /*user_data*/, TfLiteOpaqueContext* /*context*/, void* data) {
  auto* params = static_cast<StackerParams*>(data);
  if (params == nullptr) {
    return;
  }
  delete[] params->state;
  delete params;
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
  TfLiteOpaqueTensor* output_valid =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputValidTensor);
  TF_LITE_OPAQUE_ENSURE(context, output_valid != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output), 1);
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output_valid), 0);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output_valid),
                                 kTfLiteBool);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<StackerParams*>(TfLiteOpaqueNodeGetUserData(node));
  TF_LITE_OPAQUE_ENSURE(context, params != nullptr);

  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TfLiteOpaqueTensor* output_valid =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputValidTensor);

  const int16_t* input_data = GetTensorData<int16_t>(input);
  tflm_signal::CircularBufferWrite(params->circular_buffer, input_data,
                                   params->num_channels);

  // The first frame is replicated an extra left_context times to pad.
  if (!params->stacker_has_first_frame) {
    tflm_signal::CircularBufferExtend(params->circular_buffer,
                                      params->num_channels,
                                      params->stacker_left_context);
    params->stacker_has_first_frame = true;
  }

  int16_t* output_data = GetTensorData<int16_t>(output);
  bool* output_valid_data = GetTensorData<bool>(output_valid);
  if (tflm_signal::CircularBufferAvailable(params->circular_buffer) >=
      params->buffer_size) {
    tflm_signal::CircularBufferGet(params->circular_buffer, params->buffer_size,
                                   output_data);
    tflm_signal::CircularBufferDiscard(params->circular_buffer,
                                       params->step_size);
    *output_valid_data = true;
  } else {
    *output_valid_data = false;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateStackerOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r = TfLiteOperatorCreate(kTfLiteBuiltinCustom,
                                             "SignalStacker", /*version=*/1,
                                             /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(r, Init);
    TfLiteOperatorSetFreeWithData(r, Free);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_STACKER() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalStacker", CreateStackerOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
