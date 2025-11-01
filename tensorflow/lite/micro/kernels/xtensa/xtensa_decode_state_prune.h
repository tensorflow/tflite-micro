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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_PRUNE_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_PRUNE_H_

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/decode_state_prune.h"

namespace tflite {

class XtensaDecodeStatePrune : public DecodeStatePrune {
 public:
  XtensaDecodeStatePrune() = delete;

  XtensaDecodeStatePrune(const TfLiteContext* context,
                         MicroProfilerInterface* profiler)
      : DecodeStatePrune(context, profiler) {}

  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override;

 protected:
  virtual ~XtensaDecodeStatePrune() = default;

  void DecompressToBufferInt8_Xtensa(void* buffer);
  void DecompressToBufferPerChannelInt8_Xtensa(void* buffer);
  void DecompressToBufferPerChannelAltAxisInt8_Xtensa(void* buffer);
  void DecompressToBufferInt16_Xtensa(void* buffer);

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_PRUNE_H_
