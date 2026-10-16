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

#include "signal/src/irfft.h"

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

struct IrfftParams {
  int32_t fft_length = 0;
  int32_t input_size = 0;
  int32_t input_length = 0;
  int32_t output_length = 0;
  TfLiteType fft_type = kTfLiteNoType;
  int8_t* state = nullptr;
};

void* Init(void* /*user_data*/, TfLiteOpaqueContext* context,
           const char* buffer, size_t length) {
  auto* params = new IrfftParams;
  const flexbuffers::Map& m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
          .AsMap();
  params->fft_length = m["fft_length"].AsInt32();
  int32_t tensor_type = m["T"].AsInt32();
  size_t state_size = 0;

  switch (tensor_type) {
    case kTensorTypeFloat32:
      params->fft_type = kTfLiteFloat32;
      state_size = tflm_signal::IrfftFloatGetNeededMemory(params->fft_length);
      params->state = new int8_t[state_size];
      tflm_signal::IrfftFloatInit(params->fft_length, params->state,
                                  state_size);
      break;
    case kTensorTypeInt16:
      params->fft_type = kTfLiteInt16;
      state_size = tflm_signal::IrfftInt16GetNeededMemory(params->fft_length);
      params->state = new int8_t[state_size];
      tflm_signal::IrfftInt16Init(params->fft_length, params->state,
                                  state_size);
      break;
    case kTensorTypeInt32:
      params->fft_type = kTfLiteInt32;
      state_size = tflm_signal::IrfftInt32GetNeededMemory(params->fft_length);
      params->state = new int8_t[state_size];
      tflm_signal::IrfftInt32Init(params->fft_length, params->state,
                                  state_size);
      break;
    default:
      TF_LITE_OPAQUE_KERNEL_LOG(
          context, "Type %d is currently not supported by SignalIrfft.",
          tensor_type);
      delete params;
      return nullptr;
  }
  return params;
}

void Free(void* /*user_data*/, TfLiteOpaqueContext* /*context*/, void* data) {
  auto* params = static_cast<IrfftParams*>(data);
  if (params == nullptr) {
    return;
  }
  delete[] params->state;
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

  TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueTensorNumDims(input),
                           TfLiteOpaqueTensorNumDims(output));

  auto* params = static_cast<IrfftParams*>(TfLiteOpaqueNodeGetUserData(node));
  TF_LITE_OPAQUE_ENSURE(context, params != nullptr);

  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(input),
                                 params->fft_type);
  TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, TfLiteOpaqueTensorType(output),
                                 params->fft_type);

  int input_dims = TfLiteOpaqueTensorNumDims(input);
  int output_dims = TfLiteOpaqueTensorNumDims(output);
  TF_LITE_OPAQUE_ENSURE(context, input_dims >= 1);
  // Divide by 2 because input is complex pairs.
  params->input_length = TfLiteOpaqueTensorDim(input, input_dims - 1) / 2;
  params->input_size = GetFlatSize(input) / 2;
  params->output_length = TfLiteOpaqueTensorDim(output, output_dims - 1);
  return kTfLiteOk;
}

TfLiteStatus Eval(void* /*user_data*/, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node) {
  auto* params = static_cast<IrfftParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input =
      TfLiteOpaqueNodeGetInput(context, node, kInputTensor);
  TfLiteOpaqueTensor* output =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputTensor);

  for (int input_idx = 0, output_idx = 0; input_idx < params->input_size;
       input_idx += params->input_length, output_idx += params->output_length) {
    switch (params->fft_type) {
      case kTfLiteFloat32:
        tflm_signal::IrfftFloatApply(
            params->state, &GetTensorData<Complex<float>>(input)[input_idx],
            &GetTensorData<float>(output)[output_idx]);
        break;
      case kTfLiteInt16:
        tflm_signal::IrfftInt16Apply(
            params->state, &GetTensorData<Complex<int16_t>>(input)[input_idx],
            &GetTensorData<int16_t>(output)[output_idx]);
        break;
      case kTfLiteInt32:
        tflm_signal::IrfftInt32Apply(
            params->state, &GetTensorData<Complex<int32_t>>(input)[input_idx],
            &GetTensorData<int32_t>(output)[output_idx]);
        break;
      default:
        TF_LITE_OPAQUE_KERNEL_LOG(context, "Invalid Irfft Type %d.",
                                  params->fft_type);
        return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteOperator* CreateIrfftOperator() {
  static TfLiteOperator* op = []() {
    TfLiteOperator* r =
        TfLiteOperatorCreate(kTfLiteBuiltinCustom, "SignalIrfft", /*version=*/1,
                             /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(r, Init);
    TfLiteOperatorSetFreeWithData(r, Free);
    TfLiteOperatorSetPrepareWithData(r, Prepare);
    TfLiteOperatorSetInvokeWithData(r, Eval);
    return r;
  }();
  return op;
}

::TfLiteRegistration* Register_IRFFT() {
  static ::TfLiteRegistration r =
      MakeRegistration("SignalIrfft", CreateIrfftOperator());
  return &r;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
