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

#include "tensorflow/lite/micro/kernels/decode_state.h"

#include "tensorflow/lite/micro/kernels/decode_state_huffman.h"
#include "tensorflow/lite/micro/kernels/decode_state_lut.h"
#include "tensorflow/lite/micro/kernels/decode_state_prune.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {

DecodeState* DecodeState::CreateDecodeStateLUT(
    const TfLiteContext* context, MicroProfilerInterface* profiler) {
  MicroContext* const micro_context = GetMicroContext(context);
  void* buffer =
      micro_context->AllocatePersistentBuffer(sizeof(DecodeStateLut));
  if (buffer == nullptr) {
    return nullptr;
  }
  DecodeState* dsp = new (buffer) DecodeStateLut(context, profiler);

  return dsp;
}

DecodeState* DecodeState::CreateDecodeStatePrune(
    const TfLiteContext* context, MicroProfilerInterface* profiler) {
  MicroContext* const micro_context = GetMicroContext(context);
  void* buffer =
      micro_context->AllocatePersistentBuffer(sizeof(DecodeStatePrune));
  if (buffer == nullptr) {
    return nullptr;
  }
  DecodeState* dsp = new (buffer) DecodeStatePrune(context, profiler);

  return dsp;
}

DecodeState* DecodeState::CreateDecodeStateHuffman(
    const TfLiteContext* context, MicroProfilerInterface* profiler) {
  MicroContext* const micro_context = GetMicroContext(context);
  void* buffer =
      micro_context->AllocatePersistentBuffer(sizeof(DecodeStateHuffman));
  if (buffer == nullptr) {
    return nullptr;
  }
  DecodeState* dsp = new (buffer) DecodeStateHuffman(context, profiler);

  return dsp;
}

}  // namespace tflite
