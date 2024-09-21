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

  void DecompressToBufferWidth4_8(int8_t* buffer);
  void DecompressToBufferWidth4(int8_t* buffer);

  void DecompressToBufferWidth2_16(int8_t* buffer);

  template <typename T>
  void DecompressToBufferWidthAny(T* buffer);

  inline size_t GetNextTableIndex();
  inline size_t GetNextTableIndexWidth4();
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

void DecompressionState::DecompressToBufferWidth4_8(int8_t* buffer) {
  MicroProfiler* profiler =
      static_cast<MicroProfiler*>(micro_context_->external_context());
  ScopedMicroProfiler scoped_profiler(__func__, profiler);

  const uint32_t* indices =
      reinterpret_cast<const uint32_t*>(compressed_indices_);
  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    for (size_t count = 0; count < max_count; count += 8) {
      uint32_t index = *indices++;
      uint32_t value, value2;
      value = static_cast<uint32_t>(value_table[(index >> 4) & 0x0F]);
      value |= static_cast<uint32_t>(value_table[index & 0x0F]) << 8;
      value |= static_cast<uint32_t>(value_table[(index >> 12) & 0x0F]) << 16;
      value |= static_cast<uint32_t>(value_table[(index >> 8) & 0x0F]) << 24;
      value2 = static_cast<uint32_t>(value_table[(index >> 20) & 0x0F]);
      value2 |= static_cast<uint32_t>(value_table[(index >> 16) & 0x0F]) << 8;
      value2 |= static_cast<uint32_t>(value_table[(index >> 28) & 0x0F]) << 16;
      value2 |= static_cast<uint32_t>(value_table[(index >> 24) & 0x0F]) << 24;
      *reinterpret_cast<uint32_t*>(buffer) = value;
      *reinterpret_cast<uint32_t*>(buffer + 4) = value2;
      buffer += 8;
    }
    value_table += stride;
  }
}

void DecompressionState::DecompressToBufferWidth4(int8_t* buffer) {
  MicroProfiler* profiler =
      static_cast<MicroProfiler*>(micro_context_->external_context());
  ScopedMicroProfiler scoped_profiler(__func__, profiler);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const int8_t* value_table =
      static_cast<const int8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;
    while (count >= 16) {
      count -= 16;
      size_t index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];

      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
      index = GetNextTableIndexWidth4();
      *buffer++ = value_table[index];
    }
    while (count-- > 0) {
      size_t index = GetNextTableIndexWidth4();
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

    while (count > 0 && (current_offset & 0x0F)) {
      const size_t index = GetNextTableIndexWidth2(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

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

    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth2(current_offset++);
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
    if (!(elements_per_channel_ & 0x07)) {
      DecompressToBufferWidth4_8(static_cast<int8_t*>(buffer));
    } else {
      DecompressToBufferWidth4(static_cast<int8_t*>(buffer));
    }
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 2 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth2_16(static_cast<int8_t*>(buffer));
  } else {
    DecompressToBufferWidthAny<T>(static_cast<T*>(buffer));
  }

  return static_cast<T*>(buffer);
}

inline size_t DecompressionState::GetNextTableIndexWidth4() {
  if (current_offset_ & 1) {
    return compressed_indices_[current_offset_++ >> 1] & 0x0F;
  } else {
    return compressed_indices_[current_offset_++ >> 1] >> 4;
  }
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
