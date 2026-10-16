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

#include "signal/src/overlap_add.h"

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

struct OverlapAddParams {
  int32_t frame_size = 0;
  int32_t frame_step = 0;
  int32_t outer_dims = 0;
  int32_t n_frames = 0;
  TfLiteType type = kTfLiteNoType;
  std::vector<void*> state_buffers;
};

template <typename T>
void PrepareStateBuffers(OverlapAddParams* params) {
  for (int i = 0; i < params->outer_dims; ++i) {
    if (params->state_buffers[i] == nullptr) {
      params->state_buffers[i] = new T[params->frame_size];
      memset(params->state_buffers[i], 0, sizeof(T) * params->frame_size);
    }
  }
}

template <typename T>
void EvalOverlapAdd(OverlapAddParams* params, const TfLiteOpaqueTensor* input,
                    TfLiteOpaqueTensor* output) {
  const T* input_data = GetTensorData<T>(input);
  T* output_data = GetTensorData<T>(output);
  for (int i = 0; i < params->outer_dims; ++i) {
    T* buffer = static_cast<T*>(params->state_buffers[i]);
    for (int frame = 0; frame < params->n_frames; ++frame) {
      int input_index = (i * params->n_frames + frame) * params->frame_size;
      int output_index = (i * params->n_frames + frame) * params->frame_step;
      ::tflm_signal::OverlapAdd(&input_data[input_index], buffer,
                                params->frame_size, &output_data[output_index],
                                params->frame_step);
    }
  }
}

void* Init(void* /*user_data*/, TfLiteOpaqueContext* context,
           const char* buffer, size_t length) {
  auto* params = new OverlapAddParams;
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  params->frame_step = m["frame_step"].AsInt32();
  int32_t tensor_type = m["T"].AsInt32();
  switch (tensor_type) {
    case kTensorTypeFloat32:
      params->type = kTfLiteFloat32;
      break;
    case kTensorTypeInt16:
      params->type = kTfLiteInt16;
      break;
    default:
      TF_LITE_OPAQUE_KERNEL_LOG(
          context,
          "Input type %d is currently not supported by SignalOverlapAdd.",
          tensor_type);
      delete params;
      return nullptr;
  }
  return params;
}

void Free(void* /*user_data*/, TfLiteOpaqueContext* /*context*/, void* data) {
  auto* params = static_cast<OverlapAddParams*>(data);
  if (params == nullptr) {
    return;
  }
  for (void* state_buffer : params->state_buffers) {
    if (params->type == kTfLiteFloat32) {
      delete[] static_cast<float*>(state_buffer);
    } else {
      delete[] static_cast<int16_t*>(state_buffer);
    }
  }
  delete params;
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

  auto* params =
      static_cast<OverlapAddParams*>(TfLiteOpaqueNodeGetUserData(node));
  TF_LITE_OPAQUE_ENSURE(context, params != nullptr);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 params->type);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 params->type);

  int input_dims = TfLiteOpaqueTensorNumDims(input);
  int output_dims = TfLiteOpaqueTensorNumDims(output);
  TF_LITE_OPAQUE_ENSURE(context, input_dims >= 2);
  TF_LITE_OPAQUE_ENSURE_EQ(context, input_dims, output_dims + 1);

  params->frame_size = TfLiteOpaqueTensorDim(input, input_dims - 1);
  params->n_frames = TfLiteOpaqueTensorDim(input, input_dims - 2);
  params->outer_dims =
      GetFlatSize(input) / (params->frame_size * params->n_frames);
  params->state_buffers.resize(params->outer_dims, nullptr);

  if (params->type == kTfLiteFloat32) {
    PrepareStateBuffers<float>(params);
  } else {
    PrepareStateBuffers<int16_t>(params);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params =
      static_cast<OverlapAddParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  if (params->type == kTfLiteFloat32) {
    EvalOverlapAdd<float>(params, input, output);
  } else {
    EvalOverlapAdd<int16_t>(params, input, output);
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateOverlapAddOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r = TfLiteOperatorCreate(kTfLiteBuiltinCustom,
                                             "SignalOverlapAdd", /*version=*/1,
                                             /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(r, Init);
    TfLiteOperatorSetFreeWithData(r, Free);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_OVERLAP_ADD() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalOverlapAdd", CreateOverlapAddOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
