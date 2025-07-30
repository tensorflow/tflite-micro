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

#include "tensorflow/lite/micro/kernels/xtensa/xtensa_decode_state_prune.h"

#include <cstddef>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

TfLiteStatus XtensaDecodeStatePrune::Decode(const TfLiteEvalTensor& input,
                                            const TfLiteEvalTensor& ancillary,
                                            const TfLiteEvalTensor& output) {
  void* const buffer = const_cast<void*>(micro::GetTensorData<void>(&output));
  TFLITE_DCHECK(buffer != nullptr);

  switch (output.type) {
    case kTfLiteBool:
      DecompressToBufferInt8_Xtensa(buffer);
      break;
    case kTfLiteFloat32:
      DecodeStatePrune::DecompressToBuffer<int32_t>(buffer);
      break;
    case kTfLiteInt8:
      DecompressToBufferInt8_Xtensa(buffer);
      break;
    case kTfLiteInt16:
      DecompressToBufferInt16_Xtensa(buffer);
      break;
    case kTfLiteInt32:
      DecodeStatePrune::DecompressToBuffer<int32_t>(buffer);
      break;
    case kTfLiteInt64:
      DecodeStatePrune::DecompressToBuffer<int64_t>(buffer);
      break;
    default:
      MicroPrintf("unsupported tensor type %s", TfLiteTypeGetName(output.type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

void XtensaDecodeStatePrune::DecompressToBufferInt8_Xtensa(void* buffer) {
  if (num_channels_ > 1) {
    DecompressToBufferPerChannelInt8_Xtensa(buffer);
    return;
  }

  MicroPrintf(__func__);
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  ae_int8x16* p_weights = (ae_int8x16*)value_table_;
  int* __restrict p_mask32 = (int*)compressed_indices_;
  ae_valign align = AE_LA64_PP(p_weights);
  ae_int8x8 data0, data1, data2, data3;
  ae_int8x8 shfl0, shfl1, shfl2, shfl3;
  const int count = count_indices_;
  int8_t* __restrict pCoeff = static_cast<int8_t*>(buffer);
  ae_int8x8 zero = single_zero_point_;
  ae_int8x8 discarded;

  for (int i = 0; i < count >> 5; i++) {
    // unpack elements
    int mask = *p_mask32++;
    AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 0);
    AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 1);
    AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 2);
    AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 3);
    data0 = AE_SHFL8X8(data0, shfl0);
    data1 = AE_SHFL8X8(data1, shfl1);
    data2 = AE_SHFL8X8(data2, shfl2);
    data3 = AE_SHFL8X8(data3, shfl3);

    // merge <zero> into elements
    AE_MOVT8X16_L(discarded, data0, zero, data0, mask);
    AE_MOVT8X16_L(discarded, data1, zero, data1, mask >> 8);
    AE_MOVT8X16_H(discarded, data2, zero, data2, mask);
    AE_MOVT8X16_H(discarded, data3, zero, data3, mask >> 8);

    // move merged elements to output
    AE_S8X8X2_IP(data0, data1, (ae_int8x16*)pCoeff, 16);
    AE_S8X8X2_IP(data2, data3, (ae_int8x16*)pCoeff, 16);
  }

  const int count_rem = count & 0x1F;
  if (count_rem) {
    ae_valignx2 align2 = AE_ZALIGN128();
    int8_t* __restrict p_mask8 = reinterpret_cast<int8_t*>(p_mask32);

    // unpack and merge <zero> into remaining elements
    int mask = *p_mask8++;
    AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 0);
    data0 = AE_SHFL8X8(data0, shfl0);
    AE_MOVT8X16_L(discarded, data0, zero, data0, mask);
    if (count_rem > 8) {
      mask = *p_mask8++;
      AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 0);
      data1 = AE_SHFL8X8(data1, shfl1);
      AE_MOVT8X16_L(discarded, data1, zero, data1, mask);
    }
    if (count_rem > 16) {
      mask = *p_mask8++;
      AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 0);
      data2 = AE_SHFL8X8(data2, shfl2);
      AE_MOVT8X16_L(discarded, data2, zero, data2, mask);
    }
    if (count_rem > 24) {
      mask = *p_mask8++;
      AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
      data3 = AE_SHFL8X8(data3, shfl3);
      AE_MOVT8X16_L(discarded, data3, zero, data3, mask);
    }

    // move merged elements to output
    if (count_rem <= 16) {
      AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, count_rem);
    } else {
      AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
      AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff,
                     count_rem & 0xF);
    }
    AE_SA128POS_FP(align2, pCoeff);
  }
}

void XtensaDecodeStatePrune::DecompressToBufferPerChannelInt8_Xtensa(
    void* buffer) {
  if (use_alternate_axis_) {
    DecodeStatePrune::DecompressToBufferPerChannel<int8_t>(buffer);
    return;
  }

  MicroPrintf(__func__);
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  ae_int8x16* p_weights = (ae_int8x16*)value_table_;
  int* __restrict p_mask32 = (int*)compressed_indices_;
  ae_valign align = AE_LA64_PP(p_weights);
  ae_int8x8 data0, data1, data2, data3;
  ae_int8x8 shfl0, shfl1, shfl2, shfl3;
  const int count = elements_per_channel_;
  int8_t* __restrict pCoeff = static_cast<int8_t*>(buffer);
  ae_int8x8 discarded;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    ae_int8x8 zero = zero_points_[channel];

    for (int i = 0; i < count >> 5; i++) {
      // unpack elements
      int mask = *p_mask32++;
      AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 0);
      AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 1);
      AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 2);
      AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 3);
      data0 = AE_SHFL8X8(data0, shfl0);
      data1 = AE_SHFL8X8(data1, shfl1);
      data2 = AE_SHFL8X8(data2, shfl2);
      data3 = AE_SHFL8X8(data3, shfl3);

      // merge <zero> into elements
      AE_MOVT8X16_L(discarded, data0, zero, data0, mask);
      AE_MOVT8X16_L(discarded, data1, zero, data1, mask >> 8);
      AE_MOVT8X16_H(discarded, data2, zero, data2, mask);
      AE_MOVT8X16_H(discarded, data3, zero, data3, mask >> 8);

      // move merged elements to output
      AE_S8X8X2_IP(data0, data1, (ae_int8x16*)pCoeff, 16);
      AE_S8X8X2_IP(data2, data3, (ae_int8x16*)pCoeff, 16);
    }

    const int count_rem = count & 0x1F;
    if (count_rem) {
      ae_valignx2 align2 = AE_ZALIGN128();
      int8_t* __restrict p_mask8 = reinterpret_cast<int8_t*>(p_mask32);

      // unpack and merge <zero> into remaining elements
      int mask = *p_mask8++;
      AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 0);
      data0 = AE_SHFL8X8(data0, shfl0);
      AE_MOVT8X16_L(discarded, data0, zero, data0, mask);
      if (count_rem > 8) {
        mask = *p_mask8++;
        AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 0);
        data1 = AE_SHFL8X8(data1, shfl1);
        AE_MOVT8X16_L(discarded, data1, zero, data1, mask);
      }
      if (count_rem > 16) {
        mask = *p_mask8++;
        AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 0);
        data2 = AE_SHFL8X8(data2, shfl2);
        AE_MOVT8X16_L(discarded, data2, zero, data2, mask);
      }
      if (count_rem > 24) {
        mask = *p_mask8++;
        AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
        data3 = AE_SHFL8X8(data3, shfl3);
        AE_MOVT8X16_L(discarded, data3, zero, data3, mask);
      }

      // move merged elements to output
      if (count_rem <= 16) {
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, count_rem);
      } else {
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
        AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff,
                       count_rem & 0xF);
      }
      AE_SA128POS_FP(align2, pCoeff);
    }
  }
}

void XtensaDecodeStatePrune::DecompressToBufferInt16_Xtensa(void* buffer) {
  MicroPrintf(__func__);
  if (num_channels_ > 1) {
    DecodeStatePrune::DecompressToBufferPerChannel<int16_t>(buffer);
    return;
  }

  // ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);
  DecodeStatePrune::DecompressToBuffer<int16_t>(buffer);
}

}  // namespace tflite
