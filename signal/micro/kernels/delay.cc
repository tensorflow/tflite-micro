/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 */

#include <stdint.h>

#include "signal/src/circular_buffer.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kDelayLengthIndex = 0;

struct TFLMSignalFrontendDelayParams {
  int32_t frame_size;
  int32_t delay_length;
  int32_t outer_dims;

  // 🔥 optimized memory layout
  int8_t* big_buffer;
  tflm_signal::CircularBuffer** circular_buffers;
};

void* DelayInit(TfLiteContext* context, const char* buffer, size_t length) {
  auto* params = static_cast<TFLMSignalFrontendDelayParams*>(
      context->AllocatePersistentBuffer(
          context, sizeof(TFLMSignalFrontendDelayParams)));

  if (!params) return nullptr;

  FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer), length);
  params->delay_length = fbw.ElementAsInt32(kDelayLengthIndex);

  return params;
}

TfLiteStatus DelayPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);

  TF_LITE_ENSURE(context, input && output);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);

  auto* params =
      reinterpret_cast<TFLMSignalFrontendDelayParams*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  RuntimeShape shape = GetTensorShape(input);
  int innermost = shape.Dims(shape.DimensionsCount() - 1);

  params->frame_size = innermost;
  params->outer_dims = shape.FlatSize() / innermost;

  TF_LITE_ENSURE(context, params->frame_size > 0);
  TF_LITE_ENSURE(context, params->delay_length >= 0);

  size_t capacity =
      static_cast<size_t>(params->frame_size) +
      static_cast<size_t>(params->delay_length);

  TF_LITE_ENSURE(context, capacity > params->frame_size); // overflow guard

  // allocate pointer array
  params->circular_buffers =
      static_cast<tflm_signal::CircularBuffer**>(
          context->AllocatePersistentBuffer(
              context, params->outer_dims * sizeof(void*)));

  TF_LITE_ENSURE(context, params->circular_buffers != nullptr);

  // compute total memory
  size_t single_size =
      tflm_signal::CircularBufferGetNeededMemory(capacity);

  size_t total_size = single_size * params->outer_dims;

  params->big_buffer =
      static_cast<int8_t*>(context->AllocatePersistentBuffer(
          context, total_size));

  TF_LITE_ENSURE(context, params->big_buffer != nullptr);

  // init buffers
  for (int i = 0; i < params->outer_dims; ++i) {
    int8_t* slice = params->big_buffer + i * single_size;

    params->circular_buffers[i] =
        tflm_signal::CircularBufferInit(capacity, slice, single_size);

    TF_LITE_ENSURE(context, params->circular_buffers[i] != nullptr);

    tflm_signal::CircularBufferWriteZeros(
        params->circular_buffers[i], params->delay_length);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus DelayEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TFLMSignalFrontendDelayParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kOutputTensor);

  const int16_t* input_data = micro::GetTensorData<int16_t>(input);
  int16_t* output_data = micro::GetTensorData<int16_t>(output);

  const int frame = params->frame_size;

  for (int i = 0; i < params->outer_dims; ++i) {
    auto* cb = params->circular_buffers[i];

    const int16_t* in = input_data + i * frame;
    int16_t* out = output_data + i * frame;

    tflm_signal::CircularBufferWrite(cb, in, frame);
    tflm_signal::CircularBufferGet(cb, frame, out);
    tflm_signal::CircularBufferDiscard(cb, frame);
  }

  return kTfLiteOk;
}

void DelayReset(TfLiteContext* context, void* buffer) {
  auto* params = static_cast<TFLMSignalFrontendDelayParams*>(buffer);

  for (int i = 0; i < params->outer_dims; ++i) {
    auto* cb = params->circular_buffers[i];
    tflm_signal::CircularBufferReset(cb);
    tflm_signal::CircularBufferWriteZeros(cb, params->delay_length);
  }
}

}  // namespace

namespace tflm_signal {

TFLMRegistration* Register_DELAY() {
  static TFLMRegistration r =
      micro::RegisterOp(DelayInit, DelayPrepare, DelayEval, nullptr,
                        DelayReset);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite
