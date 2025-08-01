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

#include "tensorflow/lite/micro/kernels/decode_state_lut.h"
#include "tensorflow/lite/micro/micro_context.h"

#ifdef HIFI5
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_decode_state_lut.h"
#endif  // HIFI5

namespace tflite {

DecodeState* DecodeState::CreateDecodeStateLUT(
    const TfLiteContext* context, MicroProfilerInterface* profiler) {
  MicroContext* const micro_context = GetMicroContext(context);
#ifdef HIFI5
  constexpr size_t kBufferSize = sizeof(XtensaDecodeStateLUT);
#else
  constexpr size_t kBufferSize = sizeof(DecodeStateLUT);
#endif  // HIFI5
  void* buffer = micro_context->AllocatePersistentBuffer(kBufferSize);
  if (buffer == nullptr) {
    return nullptr;
  }
#ifdef HIFI5
  DecodeState* dsp = new (buffer) XtensaDecodeStateLUT(context, profiler);
#else
  DecodeState* dsp = new (buffer) DecodeStateLUT(context, profiler);
#endif  // HIFI5

  return dsp;
}

}  // namespace tflite
