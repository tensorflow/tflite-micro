/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_context.h"

#include <cstdarg>
#include <cstddef>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

int GetTensorIndex(int index, int max_size, const int* tensor_indices) {
  if (index >= 0 && index < max_size) {
    const int tensor_index = tensor_indices[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

#ifdef USE_TFLM_COMPRESSION

struct DecompressionState {
  DecompressionState() = delete;

  DecompressionState(const uint8_t* compressed_indices,
                     const size_t count_indices,
                     const CompressionTensorData& comp_data,
                     const size_t num_channels, MicroContext* micro_context)
      : compressed_indices_(compressed_indices),
        count_indices_(count_indices),
        comp_data_(comp_data),
        num_channels_(num_channels),
        micro_context_(micro_context) {}

  template <typename T>
  T* DecompressToBuffer(void* buffer);

  void DecompressToBufferWidth4_16(int8_t* buffer);
  void DecompressToBufferWidth3_32(int8_t* buffer);
  void DecompressToBufferWidth2_16(int8_t* buffer);

  template <typename T>
  void DecompressToBufferWidthAny(T* buffer);

  inline size_t GetNextTableIndex();
  inline size_t GetNextTableIndexWidth4(const size_t current_offset);
  inline size_t GetNextTableIndexWidth3(const size_t current_offset);
  inline size_t GetNextTableIndexWidth2(const size_t current_offset);

  template <typename T>
  inline void UpdateBufferAndChannelIndex();

 private:
  const uint8_t* compressed_indices_;
  const size_t count_indices_;
  const CompressionTensorData& comp_data_;
  const size_t num_channels_;
  const size_t compressed_bit_width_ =
      comp_data_.data.lut_data->compressed_bit_width;
  size_t channel_ = 0;
  size_t index_in_channel_ = 0;
  const size_t elements_per_channel_ =
      comp_data_.data.lut_data->use_alternate_axis
          ? 1
          : count_indices_ / num_channels_;
  size_t buffer_index_ = 0;
  size_t current_offset_ = 0;
  size_t current_bits_remaining_ = 8;
  uint8_t current_byte_ = compressed_indices_[0];
  MicroContext* micro_context_;
  const void* value_table_ = comp_data_.data.lut_data->value_table;
};

void DecompressionState::DecompressToBufferWidth4_16(int8_t* buffer) {
  MicroProfiler* profiler =
      static_cast<MicroProfiler*>(micro_context_->external_context());
  ScopedMicroProfiler scoped_profiler(__func__, profiler);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;
  size_t current_offset = 0;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;

    // process elements at start of channel up to next uint64_t alignment of
    // compressed_indices_
    while (count > 0 && (current_offset & 0x0F)) {
      const size_t index = GetNextTableIndexWidth4(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

    // process elements in current channel in groups of 16
    if (count >= 16) {
      const uint64_t* indices = reinterpret_cast<const uint64_t*>(
          &compressed_indices_[current_offset >> 1]);

      while (count >= 16) {
        count -= 16;
        uint64_t index = *indices++;
        uint64_t value, value2;

        value = static_cast<uint64_t>(value_table[(index >> 4) & 0x0F]);
        value |= static_cast<uint64_t>(value_table[index & 0x0F]) << 8;
        value |= static_cast<uint64_t>(value_table[(index >> 12) & 0x0F]) << 16;
        value |= static_cast<uint64_t>(value_table[(index >> 8) & 0x0F]) << 24;
        value |= static_cast<uint64_t>(value_table[(index >> 20) & 0x0F]) << 32;
        value |= static_cast<uint64_t>(value_table[(index >> 16) & 0x0F]) << 40;
        value |= static_cast<uint64_t>(value_table[(index >> 28) & 0x0F]) << 48;
        value |= static_cast<uint64_t>(value_table[(index >> 24) & 0x0F]) << 56;

        *reinterpret_cast<uint64_t*>(buffer) = value;

        value2 = static_cast<uint64_t>(value_table[(index >> 36) & 0x0F]);
        value2 |= static_cast<uint64_t>(value_table[(index >> 32) & 0x0F]) << 8;
        value2 |= static_cast<uint64_t>(value_table[(index >> 44) & 0x0F])
                  << 16;
        value2 |= static_cast<uint64_t>(value_table[(index >> 40) & 0x0F])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index >> 52) & 0x0F])
                  << 32;
        value2 |= static_cast<uint64_t>(value_table[(index >> 48) & 0x0F])
                  << 40;
        value2 |= static_cast<uint64_t>(value_table[(index >> 60) & 0x0F])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index >> 56) & 0x0F])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 8) = value2;

        buffer += 16;
      }

      current_offset =
          (reinterpret_cast<const uint8_t*>(indices) - compressed_indices_)
          << 1;
    }

    // process remaining elements in current channel
    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth4(current_offset++);
      *buffer++ = value_table[index];
    }

    value_table += stride;
  }
}

void DecompressionState::DecompressToBufferWidth2_16(int8_t* buffer) {
  MicroProfiler* profiler =
      static_cast<MicroProfiler*>(micro_context_->external_context());
  ScopedMicroProfiler scoped_profiler(__func__, profiler);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;
  size_t current_offset = 0;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;

    // process elements at start of channel up to next uint32_t alignment of
    // compressed_indices_
    while (count > 0 && (current_offset & 0x0F)) {
      const size_t index = GetNextTableIndexWidth2(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

    // process elements in current channel in groups of 16
    if (count >= 16) {
      const uint32_t* indices = reinterpret_cast<const uint32_t*>(
          &compressed_indices_[current_offset >> 2]);

      while (count >= 16) {
        count -= 16;
        uint32_t index = *indices++;
        uint64_t value, value2;

        value = static_cast<uint64_t>(value_table[(index >> 6) & 0x03]);
        value |= static_cast<uint64_t>(value_table[(index >> 4) & 0x03]) << 8;
        value |= static_cast<uint64_t>(value_table[(index >> 2) & 0x03]) << 16;
        value |= static_cast<uint64_t>(value_table[index & 0x03]) << 24;
        value |= static_cast<uint64_t>(value_table[(index >> 14) & 0x03]) << 32;
        value |= static_cast<uint64_t>(value_table[(index >> 12) & 0x03]) << 40;
        value |= static_cast<uint64_t>(value_table[(index >> 10) & 0x03]) << 48;
        value |= static_cast<uint64_t>(value_table[(index >> 8) & 0x03]) << 56;

        *reinterpret_cast<uint64_t*>(buffer) = value;

        value2 = static_cast<uint64_t>(value_table[(index >> 22) & 0x03]);
        value2 |= static_cast<uint64_t>(value_table[(index >> 20) & 0x03]) << 8;
        value2 |= static_cast<uint64_t>(value_table[(index >> 18) & 0x03])
                  << 16;
        value2 |= static_cast<uint64_t>(value_table[(index >> 16) & 0x03])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index >> 30) & 0x03])
                  << 32;
        value2 |= static_cast<uint64_t>(value_table[(index >> 28) & 0x03])
                  << 40;
        value2 |= static_cast<uint64_t>(value_table[(index >> 26) & 0x03])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index >> 24) & 0x03])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 8) = value2;

        buffer += 16;
      }

      current_offset =
          (reinterpret_cast<const uint8_t*>(indices) - compressed_indices_)
          << 2;
    }

    // process remaining elements in current channel
    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth2(current_offset++);
      *buffer++ = value_table[index];
    }

    value_table += stride;
  }
}

void DecompressionState::DecompressToBufferWidth3_32(int8_t* buffer) {
  MicroProfiler* profiler =
      static_cast<MicroProfiler*>(micro_context_->external_context());
  ScopedMicroProfiler scoped_profiler(__func__, profiler);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;
  size_t current_offset = 0;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;

    // process elements at start of channel up to next uint32_t alignment of
    // compressed_indices_
    while (count > 0 && (current_offset & 0x1F)) {
      const size_t index = GetNextTableIndexWidth3(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

    // process elements in current channel in groups of 32
    if (count >= 32) {
      const uint32_t* indices = reinterpret_cast<const uint32_t*>(
          &compressed_indices_[(current_offset >> 5) * 12]);

      while (count >= 32) {
        count -= 32;
        uint32_t index0 = *indices++;
        uint32_t index1 = *indices++;
        uint32_t index2 = *indices++;
        uint64_t value, value2;

        value = static_cast<uint64_t>(value_table[(index0 >> 5) & 0x07]);
        value |= static_cast<uint64_t>(value_table[(index0 >> 2) & 0x07]) << 8;
        value |=
            static_cast<uint64_t>(
                value_table[((index0 << 1) & 0b110) | ((index0 >> 15) & 0b1)])
            << 16;
        value |= static_cast<uint64_t>(value_table[(index0 >> 12) & 0x07])
                 << 24;
        value |= static_cast<uint64_t>(value_table[(index0 >> 9) & 0x07]) << 32;
        value |=
            static_cast<uint64_t>(
                value_table[((index0 >> 6) & 0b100) | ((index0 >> 22) & 0b11)])
            << 40;
        value |= static_cast<uint64_t>(value_table[(index0 >> 19) & 0x07])
                 << 48;
        value |= static_cast<uint64_t>(value_table[(index0 >> 16) & 0x07])
                 << 56;

        *reinterpret_cast<uint64_t*>(buffer) = value;

        value2 = static_cast<uint64_t>(value_table[(index0 >> 29) & 0x07]);
        value2 |= static_cast<uint64_t>(value_table[(index0 >> 26) & 0x07])
                  << 8;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index0 >> 23) & 0b110) | ((index1 >> 7) & 0b1)])
            << 16;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 4) & 0x07])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 1) & 0x07])
                  << 32;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index1 << 2) & 0b100) | ((index1 >> 14) & 0b11)])
            << 40;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 11) & 0x07])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 8) & 0x07])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 8) = value2;

        value = static_cast<uint64_t>(value_table[(index1 >> 21) & 0x07]);
        value |= static_cast<uint64_t>(value_table[(index1 >> 18) & 0x07]) << 8;
        value |=
            static_cast<uint64_t>(
                value_table[((index1 >> 15) & 0b110) | ((index1 >> 31) & 0b1)])
            << 16;
        value |= static_cast<uint64_t>(value_table[(index1 >> 28) & 0x07])
                 << 24;
        value |= static_cast<uint64_t>(value_table[(index1 >> 25) & 0x07])
                 << 32;
        value |=
            static_cast<uint64_t>(
                value_table[((index1 >> 22) & 0b100) | ((index2 >> 6) & 0b11)])
            << 40;
        value |= static_cast<uint64_t>(value_table[(index2 >> 3) & 0x07]) << 48;
        value |= static_cast<uint64_t>(value_table[(index2 >> 0) & 0x07]) << 56;

        *reinterpret_cast<uint64_t*>(buffer + 16) = value;

        value2 = static_cast<uint64_t>(value_table[(index2 >> 13) & 0x07]);
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 10) & 0x07])
                  << 8;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index2 >> 7) & 0b110) | ((index2 >> 23) & 0b1)])
            << 16;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 20) & 0x07])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 17) & 0x07])
                  << 32;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index2 >> 14) & 0b100) | ((index2 >> 30) & 0b11)])
            << 40;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 27) & 0x07])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 24) & 0x07])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 24) = value2;

        buffer += 32;
        current_offset += 32;
      }
    }

    // process remaining elements in current channel
    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth3(current_offset++);
      *buffer++ = value_table[index];
    }

    value_table += stride;
  }
}

template <typename T>
void DecompressionState::DecompressToBufferWidthAny(T* buffer) {
  MicroProfiler* profiler =
      static_cast<MicroProfiler*>(micro_context_->external_context());
  ScopedMicroProfiler scoped_profiler(__func__, profiler);

  while (buffer_index_ < count_indices_) {
    const size_t table_index = GetNextTableIndex();
    buffer[buffer_index_] = static_cast<const T*>(value_table_)[table_index];
    UpdateBufferAndChannelIndex<T>();
  }
}

template <typename T>
T* DecompressionState::DecompressToBuffer(void* buffer) {
#ifdef notdef
  MicroPrintf("DecompressToBuffer: %u 0x%x", count_indices_,
              elements_per_channel_ & 0x1F);
#endif  // notdef

  if (std::is_same<T, int8_t>::value &&
      comp_data_.data.lut_data->compressed_bit_width == 4 &&
      !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth4_16(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 2 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth2_16(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 3 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth3_32(static_cast<int8_t*>(buffer));
  } else {
    DecompressToBufferWidthAny<T>(static_cast<T*>(buffer));
  }

  return static_cast<T*>(buffer);
}

inline size_t DecompressionState::GetNextTableIndexWidth4(
    const size_t current_offset) {
  if (current_offset & 1) {
    return compressed_indices_[current_offset >> 1] & 0x0F;
  } else {
    return compressed_indices_[current_offset >> 1] >> 4;
  }
}

inline size_t DecompressionState::GetNextTableIndexWidth3(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 3) * 3;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b111) {
    case 0:
      return indices[0] >> 5;
    case 1:
      return (indices[0] >> 2) & 0b111;
    case 2:
      return ((indices[0] & 0b11) << 1) | (indices[1] >> 7);
    case 3:
      return (indices[1] >> 4) & 0b111;
    case 4:
      return (indices[1] >> 1) & 0b111;
    case 5:
      return ((indices[1] & 0b1) << 2) | (indices[2] >> 6);
    case 6:
      return (indices[2] >> 3) & 0b111;
    case 7:
      return indices[2] & 0b111;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecompressionState::GetNextTableIndexWidth2(
    const size_t current_offset) {
  if (current_offset & 0b10) {
    if (current_offset & 1) {
      return compressed_indices_[current_offset >> 2] & 0x03;
    } else {
      return (compressed_indices_[current_offset >> 2] >> 2) & 0x03;
    }
  } else {
    if (current_offset & 1) {
      return (compressed_indices_[current_offset >> 2] >> 4) & 0x03;
    } else {
      return (compressed_indices_[current_offset >> 2] >> 6) & 0x03;
    }
  }
}

inline size_t DecompressionState::GetNextTableIndex() {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  size_t table_index_bits_to_fill = compressed_bit_width_;
  size_t table_index = 0;

  while (table_index_bits_to_fill > 0) {
    if (current_bits_remaining_ == 0) {
      current_offset_++;
      current_byte_ = compressed_indices_[current_offset_];
      current_bits_remaining_ = 8;
    }

    const uint8_t mask_bit_count =
        std::min(table_index_bits_to_fill,
                 std::min(compressed_bit_width_, current_bits_remaining_));
    const uint8_t current_byte_mask = (1 << mask_bit_count) - 1;
    table_index <<= mask_bit_count;
    table_index |=
        (current_byte_ >> (current_bits_remaining_ - mask_bit_count)) &
        current_byte_mask;

    table_index_bits_to_fill -= mask_bit_count;
    current_bits_remaining_ -= mask_bit_count;
  }

  return table_index;
}

template <typename T>
inline void DecompressionState::UpdateBufferAndChannelIndex() {
  buffer_index_++;
  index_in_channel_++;
  if (index_in_channel_ == elements_per_channel_) {
    index_in_channel_ = 0;
    channel_++;
    value_table_ = static_cast<const T*>(value_table_) +
                   comp_data_.data.lut_data->value_table_channel_stride;
    if (channel_ == num_channels_) {
      channel_ = 0;
      value_table_ = comp_data_.data.lut_data->value_table;
    }
  }
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace

TfLiteTensor* MicroContext::AllocateTempInputTensor(const TfLiteNode* node,
                                                    int index) {
  const int tensor_index =
      GetTensorIndex(index, node->inputs->size, node->inputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempOutputTensor(const TfLiteNode* node,
                                                     int index) {
  const int tensor_index =
      GetTensorIndex(index, node->outputs->size, node->outputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempIntermediateTensor(
    const TfLiteNode* node, int index) {
  const int tensor_index = GetTensorIndex(index, node->intermediates->size,
                                          node->intermediates->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

void MicroContextReportOpError(struct TfLiteContext* context,
                               const char* format, ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

#ifdef USE_TFLM_COMPRESSION

void* MicroContext::DecompressTensorToBuffer(
    const TfLiteEvalTensor& tensor,
    const CompressionTensorData& compression_data, void* buffer) {
  TFLITE_DCHECK(compression_data.scheme == CompressionScheme::kBinQuant);
  TFLITE_DCHECK(buffer != nullptr);
  size_t count = ElementCount(*tensor.dims);
  size_t num_channels = 1;

  if (compression_data.data.lut_data->is_per_channel_quantized) {
    const size_t channel_axis =
        compression_data.data.lut_data->use_alternate_axis
            ? tensor.dims->size - 1
            : 0;
    num_channels = tensor.dims->data[channel_axis];
  }

  DecompressionState ds(static_cast<uint8_t*>(tensor.data.data), count,
                        compression_data, num_channels, this);

  switch (tensor.type) {
    case kTfLiteBool: {
      return ds.DecompressToBuffer<bool>(buffer);
    } break;
    case kTfLiteInt8: {
      return ds.DecompressToBuffer<int8_t>(buffer);
    } break;
    case kTfLiteInt16: {
      return ds.DecompressToBuffer<int16_t>(buffer);
    } break;
    case kTfLiteInt32: {
      return ds.DecompressToBuffer<int32_t>(buffer);
    } break;
    case kTfLiteInt64: {
      return ds.DecompressToBuffer<int64_t>(buffer);
    } break;
    case kTfLiteFloat32: {
      return ds.DecompressToBuffer<float>(buffer);
    } break;
    default: {
      MicroPrintf("Unsupported decompression tensor type %d", tensor.type);
    } break;
  }

  return nullptr;
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
