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

#include <vector>

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

struct DelayParams {
  int32_t frame_size = 0;
  int32_t delay_length = 0;
  int32_t outer_dims = 0;

  std::vector<std::vector<int8_t>> state_buffers;
  std::vector<tflm_signal::CircularBuffer*> circular_buffers;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new DelayParams;
  const flexbuffers::Map m = GetFlexbufferMap(buffer, length);
  params->delay_length = m["delay_length"].AsInt32();
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

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteInt16);

  auto* params = static_cast<DelayParams*>(TfLiteOpaqueNodeGetUserData(node));
  TF_LITE_OPAQUE_ENSURE(context, params != nullptr);

  int num_dims = TfLiteOpaqueTensorNumDims(input);
  TF_LITE_OPAQUE_ENSURE(context, num_dims >= 1);
  int innermost_dim = TfLiteOpaqueTensorDim(input, num_dims - 1);
  params->outer_dims = GetFlatSize(input) / innermost_dim;
  params->frame_size = innermost_dim;

  params->state_buffers.resize(params->outer_dims);
  params->circular_buffers.resize(params->outer_dims, nullptr);
  for (int i = 0; i < params->outer_dims; ++i) {
    if (params->state_buffers[i].empty()) {
      size_t capacity = params->frame_size + params->delay_length;
      size_t state_size = tflm_signal::CircularBufferGetNeededMemory(capacity);
      params->state_buffers[i].resize(state_size);
      params->circular_buffers[i] = tflm_signal::CircularBufferInit(
          capacity, params->state_buffers[i].data(), state_size);
      tflm_signal::CircularBufferWriteZeros(params->circular_buffers[i],
                                            params->delay_length);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<DelayParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  const int16_t* input_data = GetTensorData<int16_t>(input);
  int16_t* output_data = GetTensorData<int16_t>(output);

  for (int dim_index = 0, sample_index = 0; dim_index < params->outer_dims;
       ++dim_index, sample_index += params->frame_size) {
    tflm_signal::CircularBufferWrite(params->circular_buffers[dim_index],
                                     &input_data[sample_index],
                                     params->frame_size);
    tflm_signal::CircularBufferGet(params->circular_buffers[dim_index],
                                   params->frame_size,
                                   &output_data[sample_index]);
    tflm_signal::CircularBufferDiscard(params->circular_buffers[dim_index],
                                       params->frame_size);
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateDelayOperator() {
  static TfLiteOperator* op = MakeOperator(
      "SignalDelay", Init, DeleteParams<DelayParams>, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_DELAY() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalDelay", CreateDelayOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
