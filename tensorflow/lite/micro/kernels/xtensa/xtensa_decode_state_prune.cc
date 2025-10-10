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
  if (num_channels_ > 1 && zero_points_ != nullptr) {
    DecompressToBufferPerChannelInt8_Xtensa(buffer);
    return;
  }

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

  if (single_zero_point_ == 0) {
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

      // move elements to output
      AE_S8X8X2_IP(data0, data1, (ae_int8x16*)pCoeff, 16);
      AE_S8X8X2_IP(data2, data3, (ae_int8x16*)pCoeff, 16);
    }
  } else {
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
    DecompressToBufferPerChannelAltAxisInt8_Xtensa(buffer);
    return;
  }
  TFLITE_DCHECK(zero_points_ != nullptr);

  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  ae_int8x16* p_weights = (ae_int8x16*)value_table_;
  short* __restrict p_stream = (short*)compressed_indices_;
  ae_valign align = AE_LA64_PP(p_weights);
  ae_valignx2 align2 = AE_ZALIGN128();
  ae_int8x8 data0, data1, data2, data3;
  ae_int8x8 shfl0, shfl1, shfl2, shfl3;
  const int count = elements_per_channel_;
  int8_t* __restrict pCoeff = static_cast<int8_t*>(buffer);
  ae_int8x8 discarded;

  WUR_AE_BITPTR(0);
  WUR_AE_BITHEAD(0);

  AE_DBI_IP((const unsigned short*)p_stream, 16);
  AE_DBI_IP((const unsigned short*)p_stream, 16);

  for (size_t channel = 0; channel < num_channels_; channel++) {
    ae_int8x8 zero = zero_points_[channel];
    uint32_t mask_low, mask_high;

    if (zero_points_[channel] == 0) {
      for (int i = 0; i < count >> 5; i++) {
        // unpack elements
        AE_LBI_DBI_IP((unsigned short*)p_stream, mask_high, 16);
        AE_LBI_DBI_IP((unsigned short*)p_stream, mask_low, 16);
        const int mask = (mask_high << 16) | mask_low;

        AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 3);
        AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 2);
        AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 1);
        AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
        data0 = AE_SHFL8X8(data0, shfl0);
        data1 = AE_SHFL8X8(data1, shfl1);
        data2 = AE_SHFL8X8(data2, shfl2);
        data3 = AE_SHFL8X8(data3, shfl3);

        // move elements to output
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
        AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff, 16);
      }
    } else {
      for (int i = 0; i < count >> 5; i++) {
        // unpack elements
        AE_LBI_DBI_IP((unsigned short*)p_stream, mask_high, 16);
        AE_LBI_DBI_IP((unsigned short*)p_stream, mask_low, 16);
        const int mask = (mask_high << 16) | mask_low;

        AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 3);
        AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 2);
        AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 1);
        AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
        data0 = AE_SHFL8X8(data0, shfl0);
        data1 = AE_SHFL8X8(data1, shfl1);
        data2 = AE_SHFL8X8(data2, shfl2);
        data3 = AE_SHFL8X8(data3, shfl3);

        // merge <zero> into elements
        AE_MOVT8X16_H(discarded, data0, zero, data0, mask >> 8);
        AE_MOVT8X16_H(discarded, data1, zero, data1, mask);
        AE_MOVT8X16_L(discarded, data2, zero, data2, mask >> 8);
        AE_MOVT8X16_L(discarded, data3, zero, data3, mask);

        // move merged elements to output
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
        AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff, 16);
      }
    }

    const int count_rem = count & 0x1F;
    if (count_rem) {
      if (count_rem > 16) {
        AE_LBI_DBI_IP((unsigned short*)p_stream, mask_high, 16);
        AE_LB_DB_IP((unsigned short*)p_stream, mask_low, count_rem - 16);
        mask_low <<= 32 - count_rem;
      } else {
        AE_LB_DB_IP((unsigned short*)p_stream, mask_high, count_rem);
        mask_high <<= 16 - count_rem;
        mask_low = 0;
      }
      const int mask = (mask_high << 16) | mask_low;

      // unpack and merge <zero> into remaining elements
      AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 3);
      data0 = AE_SHFL8X8(data0, shfl0);
      AE_MOVT8X16_H(discarded, data0, zero, data0, mask >> 8);
      AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 2);
      data1 = AE_SHFL8X8(data1, shfl1);
      AE_MOVT8X16_H(discarded, data1, zero, data1, mask);
      AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 1);
      data2 = AE_SHFL8X8(data2, shfl2);
      AE_MOVT8X16_L(discarded, data2, zero, data2, mask >> 8);
      AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
      data3 = AE_SHFL8X8(data3, shfl3);
      AE_MOVT8X16_L(discarded, data3, zero, data3, mask);

      // move merged elements to output
      if (count_rem <= 16) {
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, count_rem);
      } else {
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
        AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff,
                       count_rem & 0xF);
      }
    }
  }
  AE_SA128POS_FP(align2, pCoeff);
}

void XtensaDecodeStatePrune::DecompressToBufferPerChannelAltAxisInt8_Xtensa(
    void* buffer) {
  TFLITE_DCHECK(zero_points_ != nullptr);

  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  ae_int8x16* p_weights = (ae_int8x16*)value_table_;
  short* __restrict p_stream = (short*)compressed_indices_;
  ae_valign align = AE_LA64_PP(p_weights);
  ae_valignx2 align2 = AE_ZALIGN128();
  ae_int8x8 data0, data1, data2, data3;
  ae_int8x8 shfl0, shfl1, shfl2, shfl3;
  int count = count_indices_ / num_channels_;
  const int max_channels = num_channels_;
  int8_t* __restrict pCoeff = static_cast<int8_t*>(buffer);
  ae_int8x8 discarded;

  WUR_AE_BITPTR(0);
  WUR_AE_BITHEAD(0);

  AE_DBI_IP((const unsigned short*)p_stream, 16);
  AE_DBI_IP((const unsigned short*)p_stream, 16);

  while (count-- > 0) {
    ae_int8x8 zero0, zero1, zero2, zero3;
    uint32_t mask_low, mask_high;
    // p_zero is always 16 byte aligned due to copy during Setup().
    int8_t* __restrict p_zero = (int8_t*)zero_points_;

    for (int i = 0; i < max_channels >> 5; i++) {
      // unpack elements
      AE_LBI_DBI_IP((unsigned short*)p_stream, mask_high, 16);
      AE_LBI_DBI_IP((unsigned short*)p_stream, mask_low, 16);
      const int mask = (mask_high << 16) | mask_low;
      AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 3);
      AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 2);
      AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 1);
      AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
      data0 = AE_SHFL8X8(data0, shfl0);
      data1 = AE_SHFL8X8(data1, shfl1);
      data2 = AE_SHFL8X8(data2, shfl2);
      data3 = AE_SHFL8X8(data3, shfl3);

      // load <zero> values
      AE_L8X8X2_IP(zero0, zero1, (ae_int8x16*)p_zero, 16);
      AE_L8X8X2_IP(zero2, zero3, (ae_int8x16*)p_zero, 16);

      // merge <zero> into elements
      AE_MOVT8X16_H(discarded, data0, zero0, data0, mask >> 8);
      AE_MOVT8X16_H(discarded, data1, zero1, data1, mask);
      AE_MOVT8X16_L(discarded, data2, zero2, data2, mask >> 8);
      AE_MOVT8X16_L(discarded, data3, zero3, data3, mask);

      // move merged elements to output
      AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
      AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff, 16);
    }

    const int count_rem = max_channels & 0x1F;
    if (count_rem) {
      if (count_rem > 16) {
        AE_LBI_DBI_IP((unsigned short*)p_stream, mask_high, 16);
        AE_LB_DB_IP((unsigned short*)p_stream, mask_low, count_rem - 16);
        mask_low <<= 32 - count_rem;
      } else {
        AE_LB_DB_IP((unsigned short*)p_stream, mask_high, count_rem);
        mask_high <<= 16 - count_rem;
        mask_low = 0;
      }
      const int mask = (mask_high << 16) | mask_low;

      // unpack remaining elements
      AE_LAVUNSQZ8X8_XP(data0, shfl0, align, p_weights, mask, 3);
      AE_LAVUNSQZ8X8_XP(data1, shfl1, align, p_weights, mask, 2);
      AE_LAVUNSQZ8X8_XP(data2, shfl2, align, p_weights, mask, 1);
      AE_LAVUNSQZ8X8_XP(data3, shfl3, align, p_weights, mask, 0);
      data0 = AE_SHFL8X8(data0, shfl0);
      data1 = AE_SHFL8X8(data1, shfl1);
      data2 = AE_SHFL8X8(data2, shfl2);
      data3 = AE_SHFL8X8(data3, shfl3);

      // load <zero> values, merge <zero> into elements and
      // move merged elements to output
      ae_valignx2 align_zero = AE_LA128_PP(p_zero);
      if (count_rem <= 16) {
        AE_LAV8X8X2_XP(zero0, zero1, align_zero, (ae_int8x16*)p_zero,
                       count_rem);
        AE_MOVT8X16_H(discarded, data0, zero0, data0, mask >> 8);
        AE_MOVT8X16_H(discarded, data1, zero1, data1, mask);
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, count_rem);
      } else {
        AE_LAV8X8X2_XP(zero0, zero1, align_zero, (ae_int8x16*)p_zero, 16);
        AE_LAV8X8X2_XP(zero2, zero3, align_zero, (ae_int8x16*)p_zero,
                       count_rem & 0xF);
        AE_MOVT8X16_H(discarded, data0, zero0, data0, mask >> 8);
        AE_MOVT8X16_H(discarded, data1, zero1, data1, mask);
        AE_MOVT8X16_L(discarded, data2, zero2, data2, mask >> 8);
        AE_MOVT8X16_L(discarded, data3, zero3, data3, mask);
        AE_SAV8X8X2_XP(data0, data1, align2, (ae_int8x16*)pCoeff, 16);
        AE_SAV8X8X2_XP(data2, data3, align2, (ae_int8x16*)pCoeff,
                       count_rem & 0xF);
      }
    }
  }
  AE_SA128POS_FP(align2, pCoeff);
}

void XtensaDecodeStatePrune::DecompressToBufferInt16_Xtensa(void* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  ae_int16x8* p_weights = (ae_int16x8*)value_table_;
  int* __restrict p_mask32 = (int*)compressed_indices_;
  ae_valign align = AE_LA64_PP(p_weights);
  ae_int16x4 data0, data1, data2, data3;
  ae_int16x4 data4, data5, data6, data7;
  ae_int16x4 shfl0, shfl1, shfl2, shfl3;
  ae_int16x4 shfl4, shfl5, shfl6, shfl7;
  const int count = count_indices_;
  int16_t* __restrict pCoeff = static_cast<int16_t*>(buffer);

  for (int i = 0; i < count >> 5; i++) {
    // unpack elements and merge 0 (zero) elements
    int mask = *p_mask32++;
    AE_LAVUNSQZ16X4_XP(data0, shfl0, align, p_weights, mask, 1);
    AE_LAVUNSQZ16X4_XP(data1, shfl1, align, p_weights, mask, 0);
    AE_LAVUNSQZ16X4_XP(data2, shfl2, align, p_weights, mask, 3);
    AE_LAVUNSQZ16X4_XP(data3, shfl3, align, p_weights, mask, 2);
    AE_LAVUNSQZ16X4_XP(data4, shfl4, align, p_weights, mask, 5);
    AE_LAVUNSQZ16X4_XP(data5, shfl5, align, p_weights, mask, 4);
    AE_LAVUNSQZ16X4_XP(data6, shfl6, align, p_weights, mask, 7);
    AE_LAVUNSQZ16X4_XP(data7, shfl7, align, p_weights, mask, 6);
    data0 = AE_SHFL16X4(data0, shfl0);
    data1 = AE_SHFL16X4(data1, shfl1);
    data2 = AE_SHFL16X4(data2, shfl2);
    data3 = AE_SHFL16X4(data3, shfl3);
    data4 = AE_SHFL16X4(data4, shfl4);
    data5 = AE_SHFL16X4(data5, shfl5);
    data6 = AE_SHFL16X4(data6, shfl6);
    data7 = AE_SHFL16X4(data7, shfl7);

    // move merged elements to output
    AE_S16X4X2_IP(data0, data1, (ae_int16x8*)pCoeff, 16);
    AE_S16X4X2_IP(data2, data3, (ae_int16x8*)pCoeff, 16);
    AE_S16X4X2_IP(data4, data5, (ae_int16x8*)pCoeff, 16);
    AE_S16X4X2_IP(data6, data7, (ae_int16x8*)pCoeff, 16);
  }

  const int count_rem = count & 0x1F;
  if (count_rem) {
    ae_valignx2 align2 = AE_ZALIGN128();
    int8_t* __restrict p_mask8 = reinterpret_cast<int8_t*>(p_mask32);

    // unpack and merge <zero> into remaining elements
    int mask = *p_mask8++;
    AE_LAVUNSQZ16X4_XP(data0, shfl0, align, p_weights, mask, 1);
    AE_LAVUNSQZ16X4_XP(data1, shfl1, align, p_weights, mask, 0);
    data0 = AE_SHFL16X4(data0, shfl0);
    data1 = AE_SHFL16X4(data1, shfl1);
    if (count_rem > 8) {
      mask = *p_mask8++;
      AE_LAVUNSQZ16X4_XP(data2, shfl2, align, p_weights, mask, 1);
      AE_LAVUNSQZ16X4_XP(data3, shfl3, align, p_weights, mask, 0);
      data2 = AE_SHFL16X4(data2, shfl2);
      data3 = AE_SHFL16X4(data3, shfl3);
    }
    if (count_rem > 16) {
      mask = *p_mask8++;
      AE_LAVUNSQZ16X4_XP(data4, shfl4, align, p_weights, mask, 1);
      AE_LAVUNSQZ16X4_XP(data5, shfl5, align, p_weights, mask, 0);
      data4 = AE_SHFL16X4(data4, shfl4);
      data5 = AE_SHFL16X4(data5, shfl5);
    }
    if (count_rem > 24) {
      mask = *p_mask8++;
      AE_LAVUNSQZ16X4_XP(data6, shfl6, align, p_weights, mask, 1);
      AE_LAVUNSQZ16X4_XP(data7, shfl7, align, p_weights, mask, 0);
      data6 = AE_SHFL16X4(data6, shfl6);
      data7 = AE_SHFL16X4(data7, shfl7);
    }

    // move merged elements to output
    if (count_rem <= 8) {
      AE_SAV16X4X2_XP(data0, data1, align2, (ae_int16x8*)pCoeff,
                      count_rem << 1);
    } else if (count_rem <= 16) {
      AE_SAV16X4X2_XP(data0, data1, align2, (ae_int16x8*)pCoeff, 16);
      AE_SAV16X4X2_XP(data2, data3, align2, (ae_int16x8*)pCoeff,
                      (count_rem - 8) << 1);
    } else if (count_rem <= 24) {
      AE_SAV16X4X2_XP(data0, data1, align2, (ae_int16x8*)pCoeff, 16);
      AE_SAV16X4X2_XP(data2, data3, align2, (ae_int16x8*)pCoeff, 16);
      AE_SAV16X4X2_XP(data4, data5, align2, (ae_int16x8*)pCoeff,
                      (count_rem - 16) << 1);
    } else {
      AE_SAV16X4X2_XP(data0, data1, align2, (ae_int16x8*)pCoeff, 16);
      AE_SAV16X4X2_XP(data2, data3, align2, (ae_int16x8*)pCoeff, 16);
      AE_SAV16X4X2_XP(data4, data5, align2, (ae_int16x8*)pCoeff, 16);
      AE_SAV16X4X2_XP(data6, data7, align2, (ae_int16x8*)pCoeff,
                      (count_rem - 24) << 1);
    }
    AE_SA128POS_FP(align2, pCoeff);
  }
}

}  // namespace tflite
