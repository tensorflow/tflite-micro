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

#include "tensorflow/lite/micro/kernels/decode_state_lut.h"

#include <cstddef>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

TfLiteStatus DecodeStateLut::Setup(const TfLiteTensor& input,
                                   const TfLiteTensor& ancillary,
                                   const TfLiteTensor& output) {
  const uint8_t* const ancillary_data = GetTensorData<uint8_t>(&ancillary);
  if (ancillary_data[kDcmVersionOffset] != 1) {
    MicroPrintf("unsupported version %u", ancillary_data[kDcmVersionOffset]);
    return kTfLiteError;
  }

  // resolve num_channels_ and use_alternate_axis_
  if (output.quantization.type == kTfLiteAffineQuantization &&
      output.quantization.params != nullptr) {
    const TfLiteAffineQuantization* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(output.quantization.params);
    num_channels_ = quantization->scale->size;
    if ((quantization->quantized_dimension == output.dims->size - 1) &&
        num_channels_ > 1) {
      use_alternate_axis_ = true;
    } else if (quantization->quantized_dimension != 0) {
      MicroPrintf("unsupported quantization axis %u",
                  quantization->quantized_dimension);
      return kTfLiteError;
    }
  }

  compressed_indices_ = GetTensorData<uint8_t>(&input);
  count_indices_ = NumElements(&output);
  elements_per_channel_ =
      use_alternate_axis_ ? 1 : count_indices_ / num_channels_;
  value_table_ = &ancillary_data[kDcmSizeInBytes];
  value_table_channel_stride_ = ancillary_data[kDcmValueTableStrideOffset];
  compressed_bit_width_ =
      ancillary_data[kDcmParamsOffset] & kDcmParamsBitWidthMask;

  return kTfLiteOk;
}

TfLiteStatus DecodeStateLut::Decode(const TfLiteEvalTensor& input,
                                    const TfLiteEvalTensor& ancillary,
                                    const TfLiteEvalTensor& output) {
  void* const buffer = const_cast<void*>(micro::GetTensorData<void>(&output));
  TFLITE_DCHECK(buffer != nullptr);

  switch (output.type) {
    case kTfLiteBool:
      DecompressToBuffer<bool>(buffer);
      break;
    case kTfLiteFloat32:
      DecompressToBuffer<float>(buffer);
      break;
    case kTfLiteInt8:
      DecompressToBuffer<int8_t>(buffer);
      break;
    case kTfLiteInt16:
      DecompressToBuffer<int16_t>(buffer);
      break;
    case kTfLiteInt32:
      DecompressToBuffer<int32_t>(buffer);
      break;
    case kTfLiteInt64:
      DecompressToBuffer<int64_t>(buffer);
      break;
    default:
      MicroPrintf("unsupported tensor type %s", TfLiteTypeGetName(output.type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

template <typename T>
T* DecodeStateLut::DecompressToBuffer(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  if (std::is_same<T, int8_t>::value && compressed_bit_width_ == 4 &&
      !use_alternate_axis_) {
    DecompressToBufferWidth4_16(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value && compressed_bit_width_ == 3 &&
             !use_alternate_axis_) {
    DecompressToBufferWidth3_32(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value && compressed_bit_width_ == 2 &&
             !use_alternate_axis_) {
    DecompressToBufferWidth2_16(static_cast<int8_t*>(buffer));
  } else {
    DecompressToBufferWidthAny<T>(static_cast<T*>(buffer));
  }

  return static_cast<T*>(buffer);
}

template bool* DecodeStateLut::DecompressToBuffer<bool>(void*);
template float* DecodeStateLut::DecompressToBuffer<float>(void*);
template int8_t* DecodeStateLut::DecompressToBuffer<int8_t>(void*);
template int16_t* DecodeStateLut::DecompressToBuffer<int16_t>(void*);
template int32_t* DecodeStateLut::DecompressToBuffer<int32_t>(void*);
template int64_t* DecodeStateLut::DecompressToBuffer<int64_t>(void*);

void DecodeStateLut::DecompressToBufferWidth4_16(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  const size_t stride = value_table_channel_stride_;
  const uint8_t* value_table = static_cast<const uint8_t*>(value_table_);
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

void DecodeStateLut::DecompressToBufferWidth2_16(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  const size_t stride = value_table_channel_stride_;
  const uint8_t* value_table = static_cast<const uint8_t*>(value_table_);
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

void DecodeStateLut::DecompressToBufferWidth3_32(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  const size_t stride = value_table_channel_stride_;
  const uint8_t* value_table = static_cast<const uint8_t*>(value_table_);
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

// TODO(ddavis-2015): templating GetNextTableIndexWidth<N> makes this method
// more than 2x faster, but with a large code size increase
template <typename T>
void DecodeStateLut::DecompressToBufferWidthAny(T* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  if (use_alternate_axis_) {
    const size_t stride = value_table_channel_stride_;
    size_t current_offset = 0;
    size_t count = count_indices_;

    while (count > 0) {
      const T* value_table = static_cast<const T*>(value_table_);
      for (size_t channel = 0; channel < num_channels_; channel++) {
        size_t index;
        switch (compressed_bit_width_) {
          case 1:
            index = GetNextTableIndexWidth1(current_offset);
            break;
          case 2:
            index = GetNextTableIndexWidth2(current_offset);
            break;
          case 3:
            index = GetNextTableIndexWidth3(current_offset);
            break;
          case 4:
            index = GetNextTableIndexWidth4(current_offset);
            break;
          case 5:
            index = GetNextTableIndexWidth5(current_offset);
            break;
          case 6:
            index = GetNextTableIndexWidth6(current_offset);
            break;
          case 7:
            index = GetNextTableIndexWidth7(current_offset);
            break;
        }
        current_offset++;
        *buffer++ = value_table[index];
        value_table += stride;
      }
      count -= num_channels_;
    }
  } else {
    const size_t stride = value_table_channel_stride_;
    const T* value_table = static_cast<const T*>(value_table_);
    const size_t max_count = elements_per_channel_;
    size_t current_offset = 0;

    for (size_t channel = 0; channel < num_channels_; channel++) {
      size_t count = max_count;

      while (count-- > 0) {
        size_t index;
        switch (compressed_bit_width_) {
          case 1:
            index = GetNextTableIndexWidth1(current_offset);
            break;
          case 2:
            index = GetNextTableIndexWidth2(current_offset);
            break;
          case 3:
            index = GetNextTableIndexWidth3(current_offset);
            break;
          case 4:
            index = GetNextTableIndexWidth4(current_offset);
            break;
          case 5:
            index = GetNextTableIndexWidth5(current_offset);
            break;
          case 6:
            index = GetNextTableIndexWidth6(current_offset);
            break;
          case 7:
            index = GetNextTableIndexWidth7(current_offset);
            break;
        }
        current_offset++;
        *buffer++ = value_table[index];
      }
      value_table += stride;
    }
  }
}

template void DecodeStateLut::DecompressToBufferWidthAny(bool*);
template void DecodeStateLut::DecompressToBufferWidthAny(float*);
template void DecodeStateLut::DecompressToBufferWidthAny(int8_t*);
template void DecodeStateLut::DecompressToBufferWidthAny(int16_t*);
template void DecodeStateLut::DecompressToBufferWidthAny(int32_t*);
template void DecodeStateLut::DecompressToBufferWidthAny(int64_t*);

inline size_t DecodeStateLut::GetNextTableIndexWidth7(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 3) * 7;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b111) {
    case 0:
      return indices[0] >> 1;
    case 1:
      return ((indices[0] & 0b1) << 6) | (indices[1] >> 2);
    case 2:
      return ((indices[1] & 0b11) << 5) | (indices[2] >> 3);
    case 3:
      return ((indices[2] & 0b111) << 4) | (indices[3] >> 4);
    case 4:
      return ((indices[3] & 0x0F) << 3) | (indices[4] >> 5);
    case 5:
      return ((indices[4] & 0x1F) << 2) | (indices[5] >> 6);
    case 6:
      return ((indices[5] & 0x3F) << 1) | (indices[6] >> 7);
    case 7:
      return indices[6] & 0x7F;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecodeStateLut::GetNextTableIndexWidth6(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 2) * 3;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b11) {
    case 0:
      return indices[0] >> 2;
    case 1:
      return ((indices[0] & 0b11) << 4) | (indices[1] >> 4);
    case 2:
      return ((indices[1] & 0x0F) << 2) | (indices[2] >> 6);
    case 3:
      return indices[2] & 0x3F;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecodeStateLut::GetNextTableIndexWidth5(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 3) * 5;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b111) {
    case 0:
      return indices[0] >> 3;
    case 1:
      return ((indices[0] & 0b111) << 2) | (indices[1] >> 6);
    case 2:
      return (indices[1] >> 1) & 0x1F;
    case 3:
      return ((indices[1] & 0b1) << 4) | (indices[2] >> 4);
    case 4:
      return ((indices[2] & 0x0F) << 1) | (indices[3] >> 7);
    case 5:
      return (indices[3] >> 2) & 0x1F;
    case 6:
      return ((indices[3] & 0b11) << 3) | (indices[4] >> 5);
    case 7:
      return indices[4] & 0x1F;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecodeStateLut::GetNextTableIndexWidth4(
    const size_t current_offset) {
  if (current_offset & 1) {
    return compressed_indices_[current_offset >> 1] & 0x0F;
  } else {
    return compressed_indices_[current_offset >> 1] >> 4;
  }
}

inline size_t DecodeStateLut::GetNextTableIndexWidth3(
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

inline size_t DecodeStateLut::GetNextTableIndexWidth2(
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

inline size_t DecodeStateLut::GetNextTableIndexWidth1(
    const size_t current_offset) {
  const size_t shift = ~current_offset & 0b111;
  return (compressed_indices_[current_offset >> 3] >> shift) & 0b1;
}

}  // namespace tflite
