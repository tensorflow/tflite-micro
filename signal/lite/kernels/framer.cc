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
constexpr int kOutputValidTensor = 1;

struct FramerParams {
  int32_t frame_size = 0;
  int32_t frame_step = 0;
  int32_t outer_dims = 0;
  int32_t n_frames = 0;
  bool prefill = false;

  std::vector<std::vector<int8_t>> state_buffers;
  std::vector<tflm_signal::CircularBuffer*> circular_buffers;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* /*context*/,
           const char* buffer, size_t length) {
  auto* params = new FramerParams;
  const flexbuffers::Map m = GetFlexbufferMap(buffer, length);
  params->frame_size = m["frame_size"].AsInt32();
  params->frame_step = m["frame_step"].AsInt32();
  params->prefill = m["prefill"].AsBool();
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
  TfLiteOpaqueTensor* output_valid =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputValidTensor);
  TF_LITE_OPAQUE_ENSURE(context, output_valid != nullptr);

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input) + 1,
                           TfLiteOpaqueTensorNumDims(output));
  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(output_valid), 0);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 kTfLiteInt16);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output_valid),
                                 kTfLiteBool);

  auto* params = static_cast<FramerParams*>(TfLiteOpaqueNodeGetUserData(node));
  int num_dims = TfLiteOpaqueTensorNumDims(input);
  TF_LITE_OPAQUE_ENSURE(context, num_dims >= 1);
  int innermost_dim = TfLiteOpaqueTensorDim(input, num_dims - 1);
  TF_LITE_OPAQUE_ENSURE(context, innermost_dim >= params->frame_step);
  TF_LITE_OPAQUE_ENSURE_EQ(context, innermost_dim % params->frame_step, 0);
  params->outer_dims = GetFlatSize(input) / innermost_dim;
  params->n_frames = innermost_dim / params->frame_step;

  params->state_buffers.resize(params->outer_dims);
  params->circular_buffers.resize(params->outer_dims, nullptr);
  for (int i = 0; i < params->outer_dims; ++i) {
    if (params->state_buffers[i].empty()) {
      size_t capacity =
          static_cast<size_t>((params->frame_size + params->frame_step - 1) /
                              params->frame_step) *
          params->frame_step;
      size_t state_size = tflm_signal::CircularBufferGetNeededMemory(capacity);
      params->state_buffers[i].resize(state_size);
      params->circular_buffers[i] = tflm_signal::CircularBufferInit(
          capacity, params->state_buffers[i].data(), state_size);
      if (params->prefill) {
        tflm_signal::CircularBufferWriteZeros(
            params->circular_buffers[i],
            params->frame_size - params->frame_step);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<FramerParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);
  TfLiteOpaqueTensor* output_valid =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputValidTensor);

  const int16_t* input_data = GetTensorData<int16_t>(input);
  int16_t* output_data = GetTensorData<int16_t>(output);
  bool* output_valid_data = GetTensorData<bool>(output_valid);
  *output_valid_data = true;

  for (int i = 0; i < params->outer_dims; ++i) {
    for (int frame = 0; frame < params->n_frames; ++frame) {
      int input_idx = (i * params->n_frames + frame) * params->frame_step;
      int output_idx = (i * params->n_frames + frame) * params->frame_size;
      tflm_signal::CircularBufferWrite(params->circular_buffers[i],
                                       &input_data[input_idx],
                                       params->frame_step);

      if (tflm_signal::CircularBufferAvailable(params->circular_buffers[i]) >=
          static_cast<size_t>(params->frame_size)) {
        tflm_signal::CircularBufferGet(params->circular_buffers[i],
                                       params->frame_size,
                                       &output_data[output_idx]);
        tflm_signal::CircularBufferDiscard(params->circular_buffers[i],
                                           params->frame_step);
      } else {
        *output_valid_data = false;
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateFramerOperator() {
  static TfLiteOperator* op = MakeOperator(
      "SignalFramer", Init, DeleteParams<FramerParams>, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_FRAMER() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalFramer", CreateFramerOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
