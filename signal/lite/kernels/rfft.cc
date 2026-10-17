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

#include "signal/src/rfft.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <vector>

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

struct RfftParams {
  int32_t fft_length = 0;
  int32_t input_size = 0;
  int32_t input_length = 0;
  int32_t output_length = 0;
  TfLiteType fft_type = kTfLiteNoType;
  std::vector<int8_t> work_area;
  std::vector<int8_t> state;
};

template <typename T>
void EvalRfft(RfftParams* params, const TfLiteOpaqueTensor* input,
              TfLiteOpaqueTensor* output,
              void (*apply_fn)(void*, const T*, Complex<T>*)) {
  const T* input_data = GetTensorData<T>(input);
  Complex<T>* output_data = GetTensorData<Complex<T>>(output);
  T* work_area = reinterpret_cast<T*>(params->work_area.data());

  for (int input_idx = 0, output_idx = 0; input_idx < params->input_size;
       input_idx += params->input_length, output_idx += params->output_length) {
    memcpy(work_area, &input_data[input_idx], sizeof(T) * params->input_length);
    // Zero-pad input to FFT length.
    memset(&work_area[params->input_length], 0,
           sizeof(T) * (params->fft_length - params->input_length));
    apply_fn(params->state.data(), work_area, &output_data[output_idx]);
  }
}

void* Init(void* /*user_data*/, TfLiteOpaqueContext* context,
           const char* buffer, size_t length) {
  std::unique_ptr<RfftParams> params(new RfftParams);
  const flexbuffers::Map m = GetFlexbufferMap(buffer, length);
  params->fft_length = m["fft_length"].AsInt32();
  int32_t tensor_type = m["T"].AsInt32();

  switch (tensor_type) {
    case kTensorTypeFloat32: {
      params->fft_type = kTfLiteFloat32;
      size_t state_size =
          ::tflm_signal::RfftFloatGetNeededMemory(params->fft_length);
      params->work_area.resize(sizeof(float) * params->fft_length);
      params->state.resize(state_size);
      ::tflm_signal::RfftFloatInit(params->fft_length, params->state.data(),
                                   state_size);
      break;
    }
    case kTensorTypeInt16: {
      params->fft_type = kTfLiteInt16;
      size_t state_size =
          ::tflm_signal::RfftInt16GetNeededMemory(params->fft_length);
      params->work_area.resize(sizeof(int16_t) * params->fft_length);
      params->state.resize(state_size);
      ::tflm_signal::RfftInt16Init(params->fft_length, params->state.data(),
                                   state_size);
      break;
    }
    case kTensorTypeInt32: {
      params->fft_type = kTfLiteInt32;
      size_t state_size =
          ::tflm_signal::RfftInt32GetNeededMemory(params->fft_length);
      params->work_area.resize(sizeof(int32_t) * params->fft_length);
      params->state.resize(state_size);
      ::tflm_signal::RfftInt32Init(params->fft_length, params->state.data(),
                                   state_size);
      break;
    }
    default:
      TF_LITE_OPAQUE_KERNEL_LOG(
          context, "Type %d is currently not supported by SignalRfft.",
          tensor_type);
      return nullptr;
  }
  return params.release();
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

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input),
                           TfLiteOpaqueTensorNumDims(output));

  auto* params = static_cast<RfftParams*>(TfLiteOpaqueNodeGetUserData(node));
  TF_LITE_OPAQUE_ENSURE(context, params != nullptr);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 params->fft_type);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 params->fft_type);

  int input_dims = TfLiteOpaqueTensorNumDims(input);
  int output_dims = TfLiteOpaqueTensorNumDims(output);
  TF_LITE_OPAQUE_ENSURE(context, input_dims >= 1);
  params->input_length = TfLiteOpaqueTensorDim(input, input_dims - 1);
  params->input_size = GetFlatSize(input);
  // Divide by 2 because output is complex pairs.
  params->output_length = TfLiteOpaqueTensorDim(output, output_dims - 1) / 2;

  TF_LITE_OPAQUE_ENSURE(context, params->input_length <= params->fft_length);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<RfftParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  switch (params->fft_type) {
    case kTfLiteFloat32:
      EvalRfft<float>(params, input, output, ::tflm_signal::RfftFloatApply);
      break;
    case kTfLiteInt16:
      EvalRfft<int16_t>(params, input, output, ::tflm_signal::RfftInt16Apply);
      break;
    case kTfLiteInt32:
      EvalRfft<int32_t>(params, input, output, ::tflm_signal::RfftInt32Apply);
      break;
    default:
      TF_LITE_OPAQUE_KERNEL_LOG(context, "Invalid Rfft Type %d.",
                                params->fft_type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateRfftOperator() {
  static TfLiteOperator* op =
      MakeOperator("SignalRfft", Init, DeleteParams<RfftParams>, Prepare, Eval);
  return op;
}

::TfLiteRegistration* Register_RFFT() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalRfft", CreateRfftOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
