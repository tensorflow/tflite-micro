/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const size_t num_inputs = NumInputs(node);
  const size_t num_outputs = NumOutputs(node);
  TF_LITE_ENSURE(context, num_outputs > 0);
  TF_LITE_ENSURE_EQ(context, num_inputs, num_outputs * 2);

  MicroContext* const micro_context = GetMicroContext(context);

  node->user_data = micro_context->AllocatePersistentBuffer(
      num_outputs * sizeof(DecodeState*));
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  DecodeState** const dsp_arr =
      reinterpret_cast<DecodeState**>(node->user_data);

  TfLiteTensor* input = nullptr;
  TfLiteTensor* ancillary = nullptr;
  TfLiteTensor* output = nullptr;
  TfLiteStatus status = kTfLiteOk;

  for (size_t i = 0; i < num_inputs; i += 2) {
    input = micro_context->AllocateTempInputTensor(node, i);
    if (input == nullptr) {
      MicroPrintf("failed to allocate input tensor %u", i);
      status = kTfLiteError;
      break;
    }
    ancillary = micro_context->AllocateTempInputTensor(node, i + 1);
    if (ancillary == nullptr) {
      MicroPrintf("failed to allocate ancillary tensor %u", i + 1);
      status = kTfLiteError;
      break;
    }
    output = micro_context->AllocateTempOutputTensor(node, i / 2);
    if (output == nullptr) {
      MicroPrintf("failed to allocate output tensor %u", i / 2);
      status = kTfLiteError;
      break;
    }

    TF_LITE_ENSURE(context, IsConstantTensor(input));
    TF_LITE_ENSURE(context, IsConstantTensor(ancillary));

    if (DecodeState::Version(*ancillary) != 1) {
      MicroPrintf("version %u != 1", DecodeState::Version(*ancillary));
      status = kTfLiteError;
      break;
    }

    DecodeState* dsp = nullptr;
    switch (DecodeState::Type(*ancillary)) {
      case DecodeState::kDcmTypeLUT:
        dsp = DecodeState::CreateDecodeStateLUT(
            context, micro_context->GetAlternateProfiler());
        break;
      case DecodeState::kDcmTypePrune:
        dsp = DecodeState::CreateDecodeStatePrune(
            context, micro_context->GetAlternateProfiler());
        break;
      case DecodeState::kDcmTypeHuffman:
        dsp = DecodeState::CreateDecodeStateHuffman(
            context, micro_context->GetAlternateProfiler());
        break;
      case DecodeState::kDcmTypeCustom:
        MicroPrintf("Custom decode type not yet supported");
        break;
      default:
        MicroPrintf("unsupported decode type %u",
                    DecodeState::Type(*ancillary));
        break;
    }

    if (dsp != nullptr) {
      status = dsp->Setup(*input, *ancillary, *output);
      if (status != kTfLiteOk) {
        break;
      }
      dsp_arr[i / 2] = dsp;
    } else {
      MicroPrintf("failed to allocate DecodeState[%u]", i / 2);
      status = kTfLiteError;
      break;
    }

    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(ancillary);
    micro_context->DeallocateTempTfLiteTensor(output);
    input = nullptr;
    ancillary = nullptr;
    output = nullptr;
  }

  if (input != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
  }
  if (ancillary != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(ancillary);
  }
  if (output != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(output);
  }

  return status;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const size_t num_inputs = NumInputs(node);
  DecodeState** const dsp_arr =
      reinterpret_cast<DecodeState**>(node->user_data);

  for (size_t i = 0; i < num_inputs; i += 2) {
    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, i);
    TF_LITE_ENSURE(context, input != nullptr);
    const TfLiteEvalTensor* ancillary =
        tflite::micro::GetEvalInput(context, node, i + 1);
    TF_LITE_ENSURE(context, ancillary != nullptr);
    const TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, i / 2);
    TF_LITE_ENSURE(context, output != nullptr);

    TfLiteStatus status = dsp_arr[i / 2]->Decode(*input, *ancillary, *output);
    TF_LITE_ENSURE(context, status == kTfLiteOk);
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_DECODE() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
