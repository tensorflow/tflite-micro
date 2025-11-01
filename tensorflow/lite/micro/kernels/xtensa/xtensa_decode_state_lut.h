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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_LUT_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_LUT_H_

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/decode_state_lut.h"

namespace tflite {

class XtensaDecodeStateLut : public DecodeStateLut {
 public:
  XtensaDecodeStateLut() = delete;

  XtensaDecodeStateLut(const TfLiteContext* context,
                       MicroProfilerInterface* profiler)
      : DecodeStateLut(context, profiler) {}

  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override;

 protected:
  virtual ~XtensaDecodeStateLut() = default;

  void DecompressToBuffer(int8_t* buffer);

  void DecompressToBufferWidth4_Xtensa(int8_t* buffer);
  void DecompressToBufferWidth3_Xtensa(int8_t* buffer);
  void DecompressToBufferWidth2_Xtensa(int8_t* buffer);

  void DecompressToBufferWidthAnyInt8_Xtensa(int8_t* buffer);
  void DecompressToBufferWidthAnyInt16_Xtensa(int16_t* buffer);
  void DecompressToBufferWidthAnyInt32_Xtensa(int32_t* buffer);
  void DecompressToBufferWidthAnyInt64_Xtensa(int64_t* buffer);

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_LUT_H_
