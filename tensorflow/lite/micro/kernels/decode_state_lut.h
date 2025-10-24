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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_LUT_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_LUT_H_

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"

namespace tflite {

class DecodeStateLut : public DecodeState {
 public:
  DecodeStateLut() = delete;

  DecodeStateLut(const TfLiteContext* context, MicroProfilerInterface* profiler)
      : DecodeState(context, profiler) {}

  virtual TfLiteStatus Setup(const TfLiteTensor& input,
                             const TfLiteTensor& ancillary,
                             const TfLiteTensor& output) override;
  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override;

 protected:
  // LUT compression constants
  static constexpr size_t kMaxBitWidth = 7;
  static constexpr size_t kMaxValueTableChannelStride = 128;

 private:
  // LUT Decode Common Metadata constants
  static constexpr size_t kDcmVersionOffset = 4;
  static constexpr size_t kDcmParamsOffset = 5;
  static constexpr uint8_t kDcmParamsBitWidthMask = 0x07;
  static constexpr size_t kDcmValueTableStrideOffset = 6;

 protected:
  virtual ~DecodeStateLut() = default;

  template <typename T>
  T* DecompressToBuffer(void* buffer);

  // optimized C++ for INT8, use_alt_axis == false
  void DecompressToBufferWidth4_16(int8_t* buffer);
  void DecompressToBufferWidth3_32(int8_t* buffer);
  void DecompressToBufferWidth2_16(int8_t* buffer);

  // generic C++ for any bit width and value table type
  template <typename T>
  void DecompressToBufferWidthAny(T* buffer);

  // Optimized C++ table index fetch
  inline size_t GetNextTableIndexWidth7(const size_t current_offset);
  inline size_t GetNextTableIndexWidth6(const size_t current_offset);
  inline size_t GetNextTableIndexWidth5(const size_t current_offset);
  inline size_t GetNextTableIndexWidth4(const size_t current_offset);
  inline size_t GetNextTableIndexWidth3(const size_t current_offset);
  inline size_t GetNextTableIndexWidth2(const size_t current_offset);
  inline size_t GetNextTableIndexWidth1(const size_t current_offset);

 protected:
  const uint8_t* compressed_indices_ = nullptr;
  size_t count_indices_ = 0;
  size_t num_channels_ = 1;
  size_t elements_per_channel_ = 0;         // computed from use_alternate_axis_
  const void* value_table_ = nullptr;       // Pointer into FlatBuffer values
  uint8_t value_table_channel_stride_ = 0;  // elements per channel
  uint8_t compressed_bit_width_ = 0;        // 1 to 7 bits
  bool use_alternate_axis_ = false;         // shape channel axis:
                                            // false = first, true = last

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_LUT_H_
