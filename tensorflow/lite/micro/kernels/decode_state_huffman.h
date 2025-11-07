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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_HUFFMAN_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_HUFFMAN_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"

namespace tflite {

class DecodeStateHuffman : public DecodeState {
 public:
  DecodeStateHuffman() = delete;

  DecodeStateHuffman(const TfLiteContext* context,
                     MicroProfilerInterface* profiler)
      : DecodeState(context, profiler) {}

  virtual TfLiteStatus Setup(const TfLiteTensor& input,
                             const TfLiteTensor& ancillary,
                             const TfLiteTensor& output) override;
  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override;

  // Huffman table element constants
  static constexpr uint16_t kTable16BitSymbolFoundMask = 0x8000;
  static constexpr uint16_t kTable16BitCountMask = 0x7800;
  static constexpr size_t kTable16BitCountShift = 11;
  static constexpr uint16_t kTable16BitValueMask = 0x07FF;
  static constexpr uint32_t kTable32BitSymbolFoundMask = 0x8000'0000;
  static constexpr uint32_t kTable32BitCountMask = 0x7800'0000;
  static constexpr size_t kTable32BitCountShift = 27;
  static constexpr uint32_t kTable32BitValueMask = 0x07FF'FFFF;

  //
  // Huffman Decode Common Metadata constants
  //
  static constexpr size_t kDcmVersionOffset = 4;
  static constexpr size_t kDcmTableSizeOffset = 5;
  // 32 bit table element if set, 16 bit otherwise
  static constexpr uint8_t kDcmTableSize32BitsMask = 0x01;
  // Initial table size of N elements where value is log2(N) - 1
  static constexpr uint8_t kDcmTableSizeInitialMask = 0xF0;
  static constexpr size_t kDcmTableSizeInitialShift = 4;

 private:
  inline bool IsLittleEndian() const {
    int i = 1;
    return (reinterpret_cast<const char*>(&i)[0] == 1);
  }

  inline uint32_t Swap32(const uint32_t x) const {
#if defined(GNUC) || defined(clang)
    return __builtin_bswap32(x);
#else
    return (x << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | (x >> 24);
#endif  // defined(GNUC) || defined(clang)
  }

 protected:
  virtual ~DecodeStateHuffman() = default;

  template <typename T>
  void DecompressToBufferWith32BitTable(T* buffer);

  void DecompressToBufferWith16BitTable(int8_t* buffer);

  void InitNextBits(uint32_t& head_offset, uint32_t& head_hold,
                    const uint32_t*& head_next) const {
    if (count_codewords_ > 0) {
      head_offset = 32;
      head_next = compressed_codewords_;
      head_hold = *head_next++;
      if (IsLittleEndian()) {
        head_hold = Swap32(head_hold);
      }
    }
  }

  inline uint32_t GetNextBits(size_t count, uint32_t& head_offset,
                              uint32_t& head_hold,
                              const uint32_t*& head_next) const {
    TFLITE_DCHECK_LE(count, 31);  // avoid 64 bit shift for <mask> below
    TFLITE_DCHECK_GT(count, 0);

    uint32_t output = 0;

    if (count > head_offset) {
      // reset head
      const uint32_t mask = (1 << head_offset) - 1;
      output = (head_hold & mask) << (count - head_offset);
      count -= head_offset;
      head_hold = *head_next++;
      if (IsLittleEndian()) {
        head_hold = Swap32(head_hold);
      }
      head_offset = 32;
    }

    const uint32_t mask = (1 << count) - 1;
    const size_t shift = head_offset - count;
    output |= (head_hold >> shift) & mask;
    head_offset -= count;

    return output;
  }

  inline void PutBackBits(size_t count, uint32_t& head_offset,
                          uint32_t& head_hold,
                          const uint32_t*& head_next) const {
    TFLITE_DCHECK_LE(count, 31);

    head_offset += count;
    if (head_offset > 32) {
      head_offset -= 32;
      head_next--;
      head_hold = *(head_next - 1);
      if (IsLittleEndian()) {
        head_hold = Swap32(head_hold);
      }
    }
  }

 protected:
  const uint32_t* compressed_codewords_ = nullptr;
  size_t count_codewords_ = 0;
  const void* huffman_tables_ = nullptr;
  bool use_32bit_table_ = false;
  uint8_t initial_table_size_ = 0;  // log2(N) - 1 where N is
                                    // number of table elements

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_HUFFMAN_H_
