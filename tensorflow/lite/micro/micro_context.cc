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

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
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
                     const size_t num_channels)
      : compressed_indices_(compressed_indices),
        count_indices_(count_indices),
        comp_data_(comp_data),
        num_channels_(num_channels) {}

  template <typename T>
  T* DecompressToBuffer(void* buffer);

  size_t GetNextTableIndex();
  void UpdateBufferAndChannelIndex();

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
};

template <typename T>
T* DecompressionState::DecompressToBuffer(void* buffer) {
  while (buffer_index_ < count_indices_) {
    const size_t table_index = GetNextTableIndex();
    static_cast<T*>(buffer)[buffer_index_] =
        static_cast<const T*>(comp_data_.data.lut_data->value_table)
            [table_index +
             (channel_ * comp_data_.data.lut_data->value_table_channel_stride)];
    UpdateBufferAndChannelIndex();
  }

  return static_cast<T*>(buffer);
}

size_t DecompressionState::GetNextTableIndex() {
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

void DecompressionState::UpdateBufferAndChannelIndex() {
  buffer_index_++;
  index_in_channel_++;
  if (index_in_channel_ == elements_per_channel_) {
    index_in_channel_ = 0;
    channel_++;
    if (channel_ == num_channels_) {
      channel_ = 0;
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

void* MicroContext::DecompressTensorToScratchBuffer(
    const TfLiteEvalTensor& tensor,
    const CompressionTensorData& compression_data, int scratch_buffer_handle) {
  TFLITE_DCHECK(compression_data.scheme == CompressionScheme::kBinQuant);
  TFLITE_DCHECK(scratch_buffer_handle != -1);
  void* scratch_buffer = GetScratchBuffer(scratch_buffer_handle);
  TFLITE_DCHECK(scratch_buffer != nullptr);
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
                        compression_data, num_channels);

  switch (tensor.type) {
    case kTfLiteBool: {
      return ds.DecompressToBuffer<bool>(scratch_buffer);
    } break;
    case kTfLiteInt8: {
      return ds.DecompressToBuffer<int8_t>(scratch_buffer);
    } break;
    case kTfLiteInt16: {
      return ds.DecompressToBuffer<int16_t>(scratch_buffer);
    } break;
    case kTfLiteInt32: {
      return ds.DecompressToBuffer<int32_t>(scratch_buffer);
    } break;
    case kTfLiteInt64: {
      return ds.DecompressToBuffer<int64_t>(scratch_buffer);
    } break;
    case kTfLiteFloat32: {
      return ds.DecompressToBuffer<float>(scratch_buffer);
    } break;
    default: {
      MicroPrintf("Unsupported decompression tensor type %d", tensor.type);
    } break;
  }

  return nullptr;
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
