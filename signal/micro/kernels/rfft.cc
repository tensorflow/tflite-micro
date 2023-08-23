/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "signal/micro/kernels/rfft.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
// 'T' is added implicitly by the TensorFlow framework when the type is resolved
// during graph construction.
// constexpr int kTypeIndex = 0;  // 'T' (unused)
constexpr int kFftLengthIndex = 1;  // 'fft_length'

template <typename T>
struct TfLiteAudioFrontendRfftParams {
  int32_t fft_length;
  int32_t input_size;
  int32_t input_length;
  int32_t output_length;
  TfLiteType fft_type;
  T* work_area;
  int8_t* state;
};

template <typename T, size_t (*get_needed_memory_func)(int32_t),
          void* (*init_func)(int32_t, void*, size_t)>
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  auto* params = static_cast<TfLiteAudioFrontendRfftParams<T>*>(
      context->AllocatePersistentBuffer(
          context, sizeof(TfLiteAudioFrontendRfftParams<T>)));

  tflite::FlexbufferWrapper fbw(buffer_t, length);
  params->fft_length = fbw.ElementAsInt32(kFftLengthIndex);
  params->fft_type = typeToTfLiteType<T>();

  params->work_area = static_cast<T*>(context->AllocatePersistentBuffer(
      context, params->fft_length * sizeof(T)));

  size_t state_size = (*get_needed_memory_func)(params->fft_length);
  params->state = static_cast<int8_t*>(
      context->AllocatePersistentBuffer(context, state_size * sizeof(int8_t)));
  (*init_func)(params->fft_length, params->state, state_size);
  return params;
}

template <typename T, TfLiteType TfLiteTypeEnum>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), NumDimensions(output));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, TfLiteTypeEnum);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, TfLiteTypeEnum);

  auto* params =
      reinterpret_cast<TfLiteAudioFrontendRfftParams<T>*>(node->user_data);
  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);
  params->input_length = input_shape.Dims(input_shape.DimensionsCount() - 1);
  params->input_size = input_shape.FlatSize();
  // Divide by 2 because output is complex.
  params->output_length =
      output_shape.Dims(output_shape.DimensionsCount() - 1) / 2;

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

template <typename T, void (*apply_func)(void*, const T* input, Complex<T>*)>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioFrontendRfftParams<T>*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);

  const T* input_data = tflite::micro::GetTensorData<T>(input);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  Complex<T>* output_data = tflite::micro::GetTensorData<Complex<T>>(output);

  for (int input_idx = 0, output_idx = 0; input_idx < params->input_size;
       input_idx += params->input_length, output_idx += params->output_length) {
    memcpy(params->work_area, &input_data[input_idx],
           sizeof(T) * params->input_length);
    // Zero pad input to FFT length
    memset(&params->work_area[params->input_length], 0,
           sizeof(T) * (params->fft_length - params->input_length));

    (*apply_func)(params->state, params->work_area, &output_data[output_idx]);
  }
  return kTfLiteOk;
}

void* InitAll(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto tensor_type = static_cast<tflite::TensorType>(m["T"].AsInt32());

  switch (tensor_type) {
    case TensorType_INT16: {
      return Init<int16_t, ::tflm_signal::RfftInt16GetNeededMemory,
                  ::tflm_signal::RfftInt16Init>(context, buffer, length);
    }
    case TensorType_INT32: {
      return Init<int32_t, ::tflm_signal::RfftInt32GetNeededMemory,
                  ::tflm_signal::RfftInt32Init>(context, buffer, length);
    }
    case TensorType_FLOAT32: {
      return Init<float, ::tflm_signal::RfftFloatGetNeededMemory,
                  ::tflm_signal::RfftFloatInit>(context, buffer, length);
    }
    default:
      return nullptr;
  }
}

TfLiteStatus PrepareAll(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioFrontendRfftParams<void>*>(node->user_data);

  switch (params->fft_type) {
    case kTfLiteInt16: {
      return Prepare<int16_t, kTfLiteInt16>(context, node);
    }
    case kTfLiteInt32: {
      return Prepare<int32_t, kTfLiteInt32>(context, node);
    }
    case kTfLiteFloat32: {
      return Prepare<float, kTfLiteFloat32>(context, node);
    }
    default:
      return kTfLiteError;
  }
}

TfLiteStatus EvalAll(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioFrontendRfftParams<void>*>(node->user_data);

  switch (params->fft_type) {
    case kTfLiteInt16: {
      return Eval<int16_t, ::tflm_signal::RfftInt16Apply>(context, node);
    }
    case kTfLiteInt32: {
      return Eval<int32_t, ::tflm_signal::RfftInt32Apply>(context, node);
    }
    case kTfLiteFloat32: {
      return Eval<float, ::tflm_signal::RfftFloatApply>(context, node);
    }
    default:
      return kTfLiteError;
  }
}
}  // namespace

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

TFLMRegistration* Register_RFFT() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(InitAll, PrepareAll, EvalAll);
  return &r;
}

TFLMRegistration* Register_RFFT_FLOAT() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      Init<float, ::tflm_signal::RfftFloatGetNeededMemory,
           ::tflm_signal::RfftFloatInit>,
      Prepare<float, kTfLiteFloat32>,
      Eval<float, ::tflm_signal::RfftFloatApply>);
  return &r;
}

TFLMRegistration* Register_RFFT_INT16() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      Init<int16_t, ::tflm_signal::RfftInt16GetNeededMemory,
           ::tflm_signal::RfftInt16Init>,
      Prepare<int16_t, kTfLiteInt16>,
      Eval<int16_t, ::tflm_signal::RfftInt16Apply>);
  return &r;
}

TFLMRegistration* Register_RFFT_INT32() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      Init<int32_t, ::tflm_signal::RfftInt32GetNeededMemory,
           ::tflm_signal::RfftInt32Init>,
      Prepare<int32_t, kTfLiteInt32>,
      Eval<int32_t, ::tflm_signal::RfftInt32Apply>);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite
