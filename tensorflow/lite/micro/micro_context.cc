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

// TODO(ddavis-2015): break this up such that template expansion is decreased
template <typename T>
T* DecompressToBuffer(const uint8_t* compressed_indices,
                      const size_t count_indices, void* buffer,
                      const CompressionTensorData& comp_data,
                      const size_t num_channels) {
  const size_t compressed_bit_width =
      comp_data.data.lut_data->compressed_bit_width;
  TFLITE_DCHECK(compressed_bit_width <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width > 0);

  size_t channel = 0;
  size_t index_in_channel = 0;
  const size_t elements_per_channel =
      comp_data.data.lut_data->use_alternate_axis
          ? 1
          : count_indices / num_channels;
  size_t buffer_index = 0;
  size_t table_index = 0;
  size_t table_index_bits_to_fill = compressed_bit_width;
  size_t current_offset = 0;
  size_t current_bits_remaining = 8;
  uint8_t current_byte = compressed_indices[current_offset];

  // no division (other than power of 2) inside loop
  while (buffer_index < count_indices) {
    while (table_index_bits_to_fill > 0) {
      if (current_bits_remaining == 0) {
        current_offset++;
        current_byte = compressed_indices[current_offset];
        current_bits_remaining = 8;
      }

      const uint8_t mask_bit_count =
          std::min(table_index_bits_to_fill,
                   std::min(compressed_bit_width, current_bits_remaining));
      const uint8_t current_byte_mask = (1 << mask_bit_count) - 1;
      table_index <<= mask_bit_count;
      table_index |=
          (current_byte >> (current_bits_remaining - mask_bit_count)) &
          current_byte_mask;

      table_index_bits_to_fill -= mask_bit_count;
      current_bits_remaining -= mask_bit_count;
    }

    static_cast<T*>(buffer)[buffer_index] =
        static_cast<const T*>(comp_data.data.lut_data->value_table)
            [table_index +
             (channel * comp_data.data.lut_data->value_table_channel_stride)];
    buffer_index++;
    table_index_bits_to_fill = compressed_bit_width;
    table_index = 0;
    index_in_channel++;
    if (index_in_channel == elements_per_channel) {
      index_in_channel = 0;
      channel++;
      if (channel == num_channels) {
        channel = 0;
      }
    }
  }

  return static_cast<T*>(buffer);
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

  switch (tensor.type) {
    case kTfLiteBool: {
      return DecompressToBuffer<bool>(static_cast<uint8_t*>(tensor.data.data),
                                      count, scratch_buffer, compression_data,
                                      num_channels);
    } break;
    case kTfLiteInt8: {
      return DecompressToBuffer<int8_t>(static_cast<uint8_t*>(tensor.data.data),
                                        count, scratch_buffer, compression_data,
                                        num_channels);
    } break;
    case kTfLiteInt16: {
      return DecompressToBuffer<int16_t>(
          static_cast<uint8_t*>(tensor.data.data), count, scratch_buffer,
          compression_data, num_channels);
    } break;
    case kTfLiteInt32: {
      return DecompressToBuffer<int32_t>(
          static_cast<uint8_t*>(tensor.data.data), count, scratch_buffer,
          compression_data, num_channels);
    } break;
    case kTfLiteInt64: {
      return DecompressToBuffer<int64_t>(
          static_cast<uint8_t*>(tensor.data.data), count, scratch_buffer,
          compression_data, num_channels);
    } break;
    case kTfLiteFloat32: {
      return DecompressToBuffer<float>(static_cast<uint8_t*>(tensor.data.data),
                                       count, scratch_buffer, compression_data,
                                       num_channels);
    } break;
    default: {
      MicroPrintf("Unsupported decompression tensor type %d", tensor.type);
    } break;
  }

  return nullptr;
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
