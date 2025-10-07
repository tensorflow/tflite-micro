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

#include "tensorflow/lite/micro/kernels/decode_state_huffman.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

TfLiteStatus DecodeStateHuffman::Setup(const TfLiteTensor& input,
                                       const TfLiteTensor& ancillary,
                                       const TfLiteTensor& output) {
  const uint8_t* const ancillary_data = GetTensorData<uint8_t>(&ancillary);
  if (ancillary_data[kDcmVersionOffset] != 1) {
    MicroPrintf("unsupported version %u", ancillary_data[kDcmVersionOffset]);
    return kTfLiteError;
  }

  compressed_codewords_ = GetTensorData<uint32_t>(&input);
  count_codewords_ = NumElements(&output);
  huffman_tables_ = &ancillary_data[kDcmSizeInBytes];
  use_32bit_table_ =
      (ancillary_data[kDcmTableSizeOffset] & kDcmTableSize32BitsMask) != 0;
  initial_table_size_ =
      (ancillary_data[kDcmTableSizeOffset] & kDcmTableSizeInitialMask) >>
      kDcmTableSizeInitialShift;

  if (!use_32bit_table_) {
    TF_LITE_ENSURE_TYPES_EQ(const_cast<TfLiteContext*>(context_), output.type,
                            kTfLiteInt8);
  }

  return kTfLiteOk;
}

TfLiteStatus DecodeStateHuffman::Decode(const TfLiteEvalTensor& input,
                                        const TfLiteEvalTensor& ancillary,
                                        const TfLiteEvalTensor& output) {
  void* const buffer = const_cast<void*>(micro::GetTensorData<void>(&output));
  TFLITE_DCHECK(buffer != nullptr);

  switch (output.type) {
    case kTfLiteInt8:
      if (use_32bit_table_) {
        DecompressToBufferWith32BitTable(static_cast<int8_t*>(buffer));
      } else {
        DecompressToBufferWith16BitTable(static_cast<int8_t*>(buffer));
      }
      break;
    case kTfLiteInt16:
      DecompressToBufferWith32BitTable(static_cast<int16_t*>(buffer));
      break;
    default:
      MicroPrintf("unsupported tensor type %s", TfLiteTypeGetName(output.type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

void DecodeStateHuffman::DecompressToBufferWith16BitTable(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  size_t remaining = count_codewords_;
  const size_t initial_table_size = initial_table_size_ + 1;
  const uint16_t* huffman_tables =
      static_cast<const uint16_t*>(huffman_tables_);
  uint32_t head_offset = 0;             // codewords bitstring state
  uint32_t head_hold = 0;               // codewords bitstring state
  const uint32_t* head_next = nullptr;  // codewords bitstring state
  uint16_t table_value = 0;

  InitNextBits(head_offset, head_hold, head_next);

  while (remaining--) {
    size_t last_used_bits = initial_table_size;
    uint32_t current_index =
        GetNextBits(last_used_bits, head_offset, head_hold, head_next);
    size_t table_offset = current_index;
    table_value = huffman_tables[table_offset];

    while (!(table_value & kTable16BitSymbolFoundMask)) {
      last_used_bits =
          ((table_value & kTable16BitCountMask) >> kTable16BitCountShift) + 1;
      current_index =
          GetNextBits(last_used_bits, head_offset, head_hold, head_next);
      const size_t next_table_offset = table_value & kTable16BitValueMask;
      table_offset += next_table_offset + current_index;
      table_value = huffman_tables[table_offset];
    }

    *buffer++ = table_value;

    const size_t symbol_residual_bits =
        (table_value & kTable16BitCountMask) >> kTable16BitCountShift;
    if (last_used_bits > symbol_residual_bits) {
      PutBackBits(last_used_bits - symbol_residual_bits, head_offset, head_hold,
                  head_next);
    }
  }
}

template <typename T>
void DecodeStateHuffman::DecompressToBufferWith32BitTable(T* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  size_t remaining = count_codewords_;
  const size_t initial_table_size = initial_table_size_ + 1;
  const uint32_t* huffman_tables =
      static_cast<const uint32_t*>(huffman_tables_);
  uint32_t head_offset = 0;             // codewords bitstring state
  uint32_t head_hold = 0;               // codewords bitstring state
  const uint32_t* head_next = nullptr;  // codewords bitstring state
  uint32_t table_value = 0;

  InitNextBits(head_offset, head_hold, head_next);

  while (remaining--) {
    size_t last_used_bits = initial_table_size;
    uint32_t current_index =
        GetNextBits(last_used_bits, head_offset, head_hold, head_next);
    size_t table_offset = current_index;
    table_value = huffman_tables[table_offset];

    while (!(table_value & kTable32BitSymbolFoundMask)) {
      last_used_bits =
          ((table_value & kTable32BitCountMask) >> kTable32BitCountShift) + 1;
      current_index =
          GetNextBits(last_used_bits, head_offset, head_hold, head_next);
      const size_t next_table_offset = table_value & kTable32BitValueMask;
      table_offset += next_table_offset + current_index;
      table_value = huffman_tables[table_offset];
    }

    *buffer++ = table_value;

    const size_t symbol_residual_bits =
        (table_value & kTable32BitCountMask) >> kTable32BitCountShift;
    if (last_used_bits > symbol_residual_bits) {
      PutBackBits(last_used_bits - symbol_residual_bits, head_offset, head_hold,
                  head_next);
    }
  }
}

template void DecodeStateHuffman::DecompressToBufferWith32BitTable<int8_t>(
    int8_t*);
template void DecodeStateHuffman::DecompressToBufferWith32BitTable<int16_t>(
    int16_t*);

}  // namespace tflite
