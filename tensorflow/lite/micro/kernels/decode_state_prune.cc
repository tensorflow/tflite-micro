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

#include "tensorflow/lite/micro/kernels/decode_state_prune.h"

#include <algorithm>
#include <cstddef>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

TfLiteStatus DecodeStatePrune::Setup(const TfLiteTensor& input,
                                     const TfLiteTensor& ancillary,
                                     const TfLiteTensor& output) {
  const uint8_t* const ancillary_data = GetTensorData<uint8_t>(&ancillary);
  if (ancillary_data[kDcmVersionOffset] != 1) {
    MicroPrintf("unsupported version %u", ancillary_data[kDcmVersionOffset]);
    return kTfLiteError;
  }

  // resolve num_channels_, use_alternate_axis_, and zero points
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

    TFLITE_DCHECK(num_channels_ ==
                  static_cast<size_t>(quantization->zero_point->size));
    bool has_non_zero_zp =
        std::any_of(quantization->zero_point->data,
                    quantization->zero_point->data + num_channels_,
                    [](int zp) { return zp != 0; });

    if (output.type != kTfLiteInt8) {
      // make sure all zero points are 0 (zero)
      TF_LITE_ENSURE_MSG(const_cast<TfLiteContext*>(context_),
                         has_non_zero_zp == false,
                         "All zero-points must be zero");
    }

    if (num_channels_ > 1 && has_non_zero_zp) {
      // copy zero points
      MicroContext* micro_context = GetMicroContext(context_);
      const size_t bufsize = num_channels_ * sizeof(*zero_points_);
      zero_points_ = static_cast<decltype(zero_points_)>(
          micro_context->AllocatePersistentBuffer(bufsize));
      if (zero_points_ == nullptr) {
        MicroPrintf("unable to allocate zero_points_");
        return kTfLiteError;
      }
      std::copy_n(quantization->zero_point->data, num_channels_, zero_points_);
    } else {
      single_zero_point_ = quantization->zero_point->data[0];
    }
  }

  compressed_indices_ = GetTensorData<uint8_t>(&input);
  count_indices_ = NumElements(&output);
  elements_per_channel_ =
      use_alternate_axis_ ? 1 : count_indices_ / num_channels_;
  value_table_ = &ancillary_data[kDcmSizeInBytes];

  return kTfLiteOk;
}

TfLiteStatus DecodeStatePrune::Decode(const TfLiteEvalTensor& input,
                                      const TfLiteEvalTensor& ancillary,
                                      const TfLiteEvalTensor& output) {
  void* const buffer = const_cast<void*>(micro::GetTensorData<void>(&output));
  TFLITE_DCHECK(buffer != nullptr);

  switch (output.type) {
    case kTfLiteBool:
      DecompressToBuffer<int8_t>(buffer);
      break;
    case kTfLiteFloat32:
      DecompressToBuffer<int32_t>(buffer);
      break;
    case kTfLiteInt8:
      if (num_channels_ > 1 && zero_points_ != nullptr) {
        DecompressToBufferPerChannelInt8(buffer);
      } else {
        DecompressToBuffer<int8_t>(buffer);
      }
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
void DecodeStatePrune::DecompressToBuffer(void* vp) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  T* buffer = static_cast<T*>(vp);
  const T* value_table = static_cast<const T*>(value_table_);
  const size_t max_count = count_indices_;
  const uint8_t* const indices = compressed_indices_;

  for (size_t index = 0; index < max_count; index++) {
    size_t shift = ~index & 0b111;
    size_t is_not_zp = (indices[index >> 3] >> shift) & 0b1;

    if (is_not_zp) {
      *buffer++ = *value_table++;
    } else {
      *buffer++ = single_zero_point_;
    }
  }
}

void DecodeStatePrune::DecompressToBufferPerChannelInt8(void* vp) {
  TFLITE_DCHECK(zero_points_ != nullptr);
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  int8_t* buffer = static_cast<int8_t*>(vp);
  size_t current_offset = 0;
  const uint8_t* const indices = compressed_indices_;
  const int8_t* value_table = static_cast<const int8_t*>(value_table_);

  if (use_alternate_axis_) {
    const size_t max_channels = num_channels_;
    size_t count = count_indices_;

    while (count > 0) {
      for (size_t channel = 0; channel < max_channels; channel++) {
        const int8_t zp = zero_points_[channel];
        size_t shift = ~current_offset & 0b111;
        size_t is_not_zp = (indices[current_offset >> 3] >> shift) & 0b1;

        if (is_not_zp) {
          *buffer++ = *value_table++;
        } else {
          *buffer++ = zp;
        }
        current_offset++;
      }
      count -= max_channels;
    }
  } else {
    const size_t max_count = elements_per_channel_;

    for (size_t channel = 0; channel < num_channels_; channel++) {
      size_t count = max_count;
      const int8_t zp = zero_points_[channel];

      while (count-- > 0) {
        size_t shift = ~current_offset & 0b111;
        size_t is_not_zp = (indices[current_offset >> 3] >> shift) & 0b1;

        if (is_not_zp) {
          *buffer++ = *value_table++;
        } else {
          *buffer++ = zp;
        }
        current_offset++;
      }
    }
  }
}

template void DecodeStatePrune::DecompressToBuffer<int8_t>(void*);
template void DecodeStatePrune::DecompressToBuffer<int16_t>(void*);
template void DecodeStatePrune::DecompressToBuffer<int32_t>(void*);
template void DecodeStatePrune::DecompressToBuffer<int64_t>(void*);

}  // namespace tflite
