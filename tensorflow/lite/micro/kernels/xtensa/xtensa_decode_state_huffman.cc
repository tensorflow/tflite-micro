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

#include "tensorflow/lite/micro/kernels/xtensa/xtensa_decode_state_huffman.h"

#include <cstddef>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

TfLiteStatus XtensaDecodeStateHuffman::Decode(const TfLiteEvalTensor& input,
                                              const TfLiteEvalTensor& ancillary,
                                              const TfLiteEvalTensor& output) {
  void* const buffer = const_cast<void*>(micro::GetTensorData<void>(&output));
  TFLITE_DCHECK(buffer != nullptr);

  switch (output.type) {
    case kTfLiteInt8:
      if (use_32bit_table_) {
        Decompress32BitTable_Xtensa(static_cast<int8_t*>(buffer));
      } else {
        Decompress16BitTable_Xtensa(static_cast<int8_t*>(buffer));
      }
      break;
    case kTfLiteInt16:
      Decompress32BitTable_Xtensa(static_cast<int16_t*>(buffer));
      break;
    default:
      MicroPrintf("unsupported tensor type %s", TfLiteTypeGetName(output.type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

void XtensaDecodeStateHuffman::Decompress16BitTable_Xtensa(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  size_t remaining = count_codewords_;
  const uint16_t* huffman_tables =
      static_cast<const uint16_t*>(huffman_tables_);
  const uint16_t* __restrict p_stream =
      reinterpret_cast<const uint16_t*>(compressed_codewords_);

  WAE_BITPTR(15);
  WAE_BITSUSED(1);
  // byte swap the preload half-word
  WAE_BITHEAD(p_stream[0] << 8 | p_stream[0] >> 8);
  WAE_SEARCHDONE(1);
  WAE_FIRST_TS(initial_table_size_);
  AE_VLDL16C(p_stream);

  while (remaining--) {
    xtbool complete = 0;
    unsigned long int symbol;

    while (!complete) {
      AE_VLDL16T(complete, symbol, huffman_tables);
      AE_VLDL16C(p_stream);
    }

    *buffer++ = symbol;
  }
}

template <typename T>
void XtensaDecodeStateHuffman::Decompress32BitTable_Xtensa(T* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  size_t remaining = count_codewords_;
  const uint32_t* huffman_tables =
      static_cast<const uint32_t*>(huffman_tables_);
  const uint16_t* __restrict p_stream =
      reinterpret_cast<const uint16_t*>(compressed_codewords_);

  WAE_BITPTR(15);
  WAE_BITSUSED(1);
  // byte swap the preload half-word
  WAE_BITHEAD(p_stream[0] << 8 | p_stream[0] >> 8);
  WAE_SEARCHDONE(1);
  WAE_FIRST_TS(initial_table_size_);
  AE_VLDL16C(p_stream);

  while (remaining--) {
    xtbool complete = 0;
    unsigned long int symbol;

    while (!complete) {
      AE_VLDL32T(complete, symbol, huffman_tables);
      AE_VLDL16C(p_stream);
    }

    *buffer++ = symbol;
  }
}

template void XtensaDecodeStateHuffman::Decompress32BitTable_Xtensa<int8_t>(
    int8_t*);
template void XtensaDecodeStateHuffman::Decompress32BitTable_Xtensa<int16_t>(
    int16_t*);

}  // namespace tflite
