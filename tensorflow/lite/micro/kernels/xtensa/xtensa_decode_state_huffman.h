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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_HUFFMAN_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_HUFFMAN_H_

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/decode_state_huffman.h"

namespace tflite {

class XtensaDecodeStateHuffman : public DecodeStateHuffman {
 public:
  XtensaDecodeStateHuffman() = delete;

  XtensaDecodeStateHuffman(const TfLiteContext* context,
                           MicroProfilerInterface* profiler)
      : DecodeStateHuffman(context, profiler) {}

  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override;

 protected:
  virtual ~XtensaDecodeStateHuffman() = default;

  template <typename T>
  void Decompress32BitTable_Xtensa(T* buffer);

  void Decompress16BitTable_Xtensa(int8_t* buffer);

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_XTENSA_DECODE_STATE_HUFFMAN_H_
