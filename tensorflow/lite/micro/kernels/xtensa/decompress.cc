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

#ifdef USE_TFLM_COMPRESSION

#include "tensorflow/lite/micro/kernels/decompress.h"

#include <cstddef>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

#ifdef HIFI5
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#endif  // HIFI5

namespace tflite {
namespace {

#ifdef HIFI5

struct DecompressionStateXtensa : DecompressionState {
  DecompressionStateXtensa() = delete;

  DecompressionStateXtensa(const DecompressionState& other)
      : DecompressionState(other) {}

  void DecompressToBufferWidth4_Xtensa(int8_t* buffer);
  void DecompressToBufferWidth4_Xtensa_Old(int8_t* buffer);

  void DecompressToBufferWidthAnyInt8_Xtensa(int8_t* buffer);
  void DecompressToBufferWidthAnyInt16_Xtensa(int16_t* buffer);
  void DecompressToBufferWidthAnyInt32_Xtensa(int32_t* buffer);
  void DecompressToBufferWidthAnyInt64_Xtensa(int64_t* buffer);
};

// TODO(ddavis-2015): unaligned/stride code has error, method not currently
// used.
void DecompressionStateXtensa::DecompressToBufferWidth4_Xtensa(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  ae_int8x8 d_shuffle_t = AE_MOVINT8X8_FROMINT64(0xFB73EA62D951C840LL);
  ae_int8x8 d_shuffle_value_t = AE_MOVINT8X8_FROMINT64(0x08192A3B4C5D6E7FLL);
  int elements_per_channel_t_by_4 = elements_per_channel_ >> 4;
  int elements_per_channel_t_rem = elements_per_channel_ & 0xF;
  int j;

  ae_int8x8 d_out1, d_out2;
  ae_int8x8 d_value_0_t, d_value_1_t;
  ae_int8x8 d_value_0, d_value_1;
  ae_int8x8 d_index, d_dummy;

  ae_int8x8* __restrict pIn_tmp = (ae_int8x8*)compressed_indices_;
  ae_int8* __restrict p_out_tmp = (ae_int8*)buffer;

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* __restrict value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);

  const uint8_t* __restrict value_table_t = value_table;

  for (size_t i = 0; i < num_channels_; i++) {
    value_table_t = value_table;
    ae_valignx2 align_vtab = AE_LA128_PP(value_table_t);
    AE_LA8X8X2_IP(d_value_0_t, d_value_1_t, align_vtab,
                  (ae_int8x16*)value_table_t);
    AE_DSEL8X8(d_value_0, d_value_1, d_value_0_t, d_value_1_t,
               d_shuffle_value_t);

    ae_valignx2 align_store = AE_ZALIGN128();
    ae_valign align_load = AE_LA64_PP(pIn_tmp);

    for (j = 0; j < elements_per_channel_t_by_4; j++) {
      AE_LA8X8_IP(d_index, align_load, pIn_tmp);
      AE_DSEL8X8(d_out1, d_out2, d_value_0, d_value_1, d_index);
      AE_DSEL8X8(d_out1, d_out2, d_out1, d_out2, d_shuffle_t);
      AE_SA8X8X2_IP(d_out1, d_out2, align_store, (ae_int8x16*)p_out_tmp);
    }

    value_table += stride;

    ae_valignx2 align_index = AE_LA128_PP(pIn_tmp);
    AE_LAV8X8X2_XP(d_index, d_dummy, align_index, (ae_int8x16*)pIn_tmp,
                   (elements_per_channel_t_rem >>
                    1)); /* Loading 48 bits for decoding 16 weight values */
    AE_DSEL8X8(d_out1, d_out2, d_value_0, d_value_1, d_index);
    AE_DSEL8X8(d_out1, d_out2, d_out1, d_out2, d_shuffle_t);
    AE_SAV8X8X2_XP(d_out1, d_out2, align_store, (ae_int8x16*)p_out_tmp,
                   elements_per_channel_t_rem);
    AE_SA128POS_FP(align_store, (ae_int8x16*)p_out_tmp);
  }
}

void DecompressionStateXtensa::DecompressToBufferWidth4_Xtensa_Old(
    int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  char shuffle_pattern_1[8] = {0x08, 0x19, 0x2A, 0x3B, 0x4C, 0x5D, 0x6E, 0x7F};
  ae_int8x8 d_shuffle_t = *(ae_int8x8*)&shuffle_pattern_1[0];

  char shuffle_pattern_2[8] = {0xFB, 0x73, 0xEA, 0x62, 0xD9, 0x51, 0xC8, 0x40};
  ae_int8x8 d_d_shuffle_t2 = *(ae_int8x8*)&shuffle_pattern_2[0];

  ae_int8x8 d_out1, d_out2;
  ae_int8x8 d_value_0, d_value_1;
  ae_int8x8 d_index;

  int elements_per_channel_t = elements_per_channel_;
  int num_channels_t = num_channels_;
  ae_int8x8* __restrict pIn_tmp = (ae_int8x8*)compressed_indices_;
  ae_int8* __restrict p_out_tmp = (ae_int8*)buffer;

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* __restrict value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);

  for (int i = 0; i < num_channels_t; i++) {
    ae_int8x8 d_value_0_t = *(ae_int8x8*)&value_table[0];
    ae_int8x8 d_value_1_t = *(ae_int8x8*)&value_table[8];

    AE_DSEL8X8(d_value_0, d_value_1, d_value_0_t, d_value_1_t, d_shuffle_t);

    for (int j = 0; j < elements_per_channel_t; j += 16) {
      AE_L8X8_IP(d_index, pIn_tmp, 8);
      AE_DSEL8X8(d_out1, d_out2, d_value_0, d_value_1, d_index);
      AE_DSEL8X8(d_out1, d_out2, d_out1, d_out2, d_d_shuffle_t2);
      AE_S8X8X2_IP(d_out1, d_out2, (ae_int8x16*)p_out_tmp, 16);
    }

    value_table += stride;
  }
}

void DecompressionStateXtensa::DecompressToBufferWidthAnyInt8_Xtensa(
    int8_t* buffer) {
  const char* func_name_p = nullptr;
  if (micro_profiler_ != nullptr) {
    static char func_name[42];
    MicroSnprintf(func_name, sizeof(func_name), "%s_%u", __func__,
                  compressed_bit_width_);
    func_name_p = func_name;
  }
  ScopedMicroProfiler scoped_profiler(func_name_p, micro_profiler_);

  const int stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* __restrict value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);

  int num_channels_t = num_channels_;
  short* __restrict p_stream = (short*)compressed_indices_;
  uint32_t index;
  ae_int8* __restrict p_out_tmp = (ae_int8*)buffer;
  const size_t bw = compressed_bit_width_;

  WUR_AE_BITPTR(0);
  WUR_AE_BITHEAD(0);

  AE_DBI_IP((const unsigned short*)p_stream, 16);
  AE_DBI_IP((const unsigned short*)p_stream, 16);

  if (comp_data_.data.lut_data->use_alternate_axis) {
    int count = count_indices_;
    const uint8_t* __restrict value_table_t = value_table;

    while (count > 0) {
      value_table = value_table_t;

      for (int channel = 0; channel < num_channels_t; channel++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int8x8 d_tmp = AE_L8_X((const ae_int8*)value_table, index);
        AE_S8_0_IP(d_tmp, p_out_tmp, 1);
        value_table += stride;
      }

      count -= num_channels_t;
    }
  } else {
    int elements_per_channel_t = elements_per_channel_;

    for (int i = 0; i < num_channels_t; i++) {
      for (int j = 0; j < elements_per_channel_t; j++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int8x8 d_tmp = AE_L8_X((const ae_int8*)value_table, index);
        AE_S8_0_IP(d_tmp, p_out_tmp, 1);
      }

      value_table += stride;
    }
  }
}

void DecompressionStateXtensa::DecompressToBufferWidthAnyInt16_Xtensa(
    int16_t* buffer) {
  const char* func_name_p = nullptr;
  if (micro_profiler_ != nullptr) {
    static char func_name[43];
    MicroSnprintf(func_name, sizeof(func_name), "%s_%u", __func__,
                  compressed_bit_width_);
    func_name_p = func_name;
  }
  ScopedMicroProfiler scoped_profiler(func_name_p, micro_profiler_);

  const int stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint16_t* __restrict value_table =
      static_cast<const uint16_t*>(comp_data_.data.lut_data->value_table);

  int num_channels_t = num_channels_;
  short* __restrict p_stream = (short*)compressed_indices_;
  uint32_t index;
  ae_int16* __restrict p_out_tmp = (ae_int16*)buffer;
  const size_t bw = compressed_bit_width_;

  WUR_AE_BITPTR(0);
  WUR_AE_BITHEAD(0);

  AE_DBI_IP((const unsigned short*)p_stream, 16);
  AE_DBI_IP((const unsigned short*)p_stream, 16);

  if (comp_data_.data.lut_data->use_alternate_axis) {
    int count = count_indices_;
    const uint16_t* __restrict value_table_t = value_table;

    while (count > 0) {
      value_table = value_table_t;

      for (int channel = 0; channel < num_channels_t; channel++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int16x4 d_tmp = AE_L16_X((const ae_int16*)value_table, index << 1);
        AE_S16_0_IP(d_tmp, p_out_tmp, 2);
        value_table += stride;
      }

      count -= num_channels_t;
    }
  } else {
    int elements_per_channel_t = elements_per_channel_;

    for (int i = 0; i < num_channels_t; i++) {
      for (int j = 0; j < elements_per_channel_t; j++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int16x4 d_tmp = AE_L16_X((const ae_int16*)value_table, index << 1);
        AE_S16_0_IP(d_tmp, p_out_tmp, 2);
      }

      value_table += stride;
    }
  }
}

void DecompressionStateXtensa::DecompressToBufferWidthAnyInt32_Xtensa(
    int32_t* buffer) {
  const char* func_name_p = nullptr;
  if (micro_profiler_ != nullptr) {
    static char func_name[43];
    MicroSnprintf(func_name, sizeof(func_name), "%s_%u", __func__,
                  compressed_bit_width_);
    func_name_p = func_name;
  }
  ScopedMicroProfiler scoped_profiler(func_name_p, micro_profiler_);

  const int stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint32_t* __restrict value_table =
      static_cast<const uint32_t*>(comp_data_.data.lut_data->value_table);

  int num_channels_t = num_channels_;
  short* __restrict p_stream = (short*)compressed_indices_;
  uint32_t index;
  ae_int32* __restrict p_out_tmp = (ae_int32*)buffer;
  const size_t bw = compressed_bit_width_;

  WUR_AE_BITPTR(0);
  WUR_AE_BITHEAD(0);

  AE_DBI_IP((const unsigned short*)p_stream, 16);
  AE_DBI_IP((const unsigned short*)p_stream, 16);

  if (comp_data_.data.lut_data->use_alternate_axis) {
    int count = count_indices_;
    const uint32_t* __restrict value_table_t = value_table;

    while (count > 0) {
      value_table = value_table_t;

      for (int channel = 0; channel < num_channels_t; channel++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int32x2 d_tmp = AE_L32_X((const ae_int32*)value_table, index << 2);
        AE_S32_L_IP(d_tmp, p_out_tmp, 4);
        value_table += stride;
      }

      count -= num_channels_t;
    }
  } else {
    int elements_per_channel_t = elements_per_channel_;

    for (int i = 0; i < num_channels_t; i++) {
      for (int j = 0; j < elements_per_channel_t; j++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int32x2 d_tmp = AE_L32_X((const ae_int32*)value_table, index << 2);
        AE_S32_L_IP(d_tmp, p_out_tmp, 4);
      }

      value_table += stride;
    }
  }
}

void DecompressionStateXtensa::DecompressToBufferWidthAnyInt64_Xtensa(
    int64_t* buffer) {
  const char* func_name_p = nullptr;
  if (micro_profiler_ != nullptr) {
    static char func_name[43];
    MicroSnprintf(func_name, sizeof(func_name), "%s_%u", __func__,
                  compressed_bit_width_);
    func_name_p = func_name;
  }
  ScopedMicroProfiler scoped_profiler(func_name_p, micro_profiler_);

  const int stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint64_t* __restrict value_table =
      static_cast<const uint64_t*>(comp_data_.data.lut_data->value_table);

  int num_channels_t = num_channels_;
  short* __restrict p_stream = (short*)compressed_indices_;
  uint32_t index;
  ae_int64* __restrict p_out_tmp = (ae_int64*)buffer;
  const size_t bw = compressed_bit_width_;

  WUR_AE_BITPTR(0);
  WUR_AE_BITHEAD(0);

  AE_DBI_IP((const unsigned short*)p_stream, 16);
  AE_DBI_IP((const unsigned short*)p_stream, 16);

  if (comp_data_.data.lut_data->use_alternate_axis) {
    int count = count_indices_;
    const uint64_t* __restrict value_table_t = value_table;

    while (count > 0) {
      value_table = value_table_t;

      for (int channel = 0; channel < num_channels_t; channel++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int64 d_tmp = AE_L64_X((const ae_int64*)value_table, index << 3);
        AE_S64_IP(d_tmp, p_out_tmp, 8);
        value_table += stride;
      }

      count -= num_channels_t;
    }
  } else {
    int elements_per_channel_t = elements_per_channel_;

    for (int i = 0; i < num_channels_t; i++) {
      for (int j = 0; j < elements_per_channel_t; j++) {
        AE_LB_DB_IP((unsigned short*)p_stream, index, bw);
        ae_int64 d_tmp = AE_L64_X((const ae_int64*)value_table, index << 3);
        AE_S64_IP(d_tmp, p_out_tmp, 8);
      }

      value_table += stride;
    }
  }
}

#endif  // HIFI5

}  // namespace

#ifdef HIFI5

template <>
bool* DecompressionState::DecompressToBuffer<bool>(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  DecompressionStateXtensa dsx(*this);

  dsx.DecompressToBufferWidthAnyInt8_Xtensa(static_cast<int8_t*>(buffer));

  return static_cast<bool*>(buffer);
}

template <>
int8_t* DecompressionState::DecompressToBuffer<int8_t>(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  DecompressionStateXtensa dsx(*this);

  if (comp_data_.data.lut_data->compressed_bit_width == 4 &&
      !comp_data_.data.lut_data->use_alternate_axis) {
    if (!(elements_per_channel_ & 0x0F) &&
        comp_data_.data.lut_data->value_table_channel_stride == 16) {
      dsx.DecompressToBufferWidth4_Xtensa_Old(static_cast<int8_t*>(buffer));
    } else {
      dsx.DecompressToBufferWidth4_16(static_cast<int8_t*>(buffer));
    }
  } else if (comp_data_.data.lut_data->compressed_bit_width == 3 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    // TODO(ddavis-2015): placeholder
    dsx.DecompressToBufferWidthAnyInt8_Xtensa(static_cast<int8_t*>(buffer));
  } else if (comp_data_.data.lut_data->compressed_bit_width == 2 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    // TODO(ddavis-2015): placeholder
    dsx.DecompressToBufferWidthAnyInt8_Xtensa(static_cast<int8_t*>(buffer));
  } else {
    dsx.DecompressToBufferWidthAnyInt8_Xtensa(static_cast<int8_t*>(buffer));
  }

  return static_cast<int8_t*>(buffer);
}

template <>
int16_t* DecompressionState::DecompressToBuffer<int16_t>(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  DecompressionStateXtensa dsx(*this);

  dsx.DecompressToBufferWidthAnyInt16_Xtensa(static_cast<int16_t*>(buffer));

  return static_cast<int16_t*>(buffer);
}

template <>
int32_t* DecompressionState::DecompressToBuffer<int32_t>(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  DecompressionStateXtensa dsx(*this);

  dsx.DecompressToBufferWidthAnyInt32_Xtensa(static_cast<int32_t*>(buffer));

  return static_cast<int32_t*>(buffer);
}

template <>
float* DecompressionState::DecompressToBuffer<float>(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  DecompressionStateXtensa dsx(*this);

  dsx.DecompressToBufferWidthAnyInt32_Xtensa(static_cast<int32_t*>(buffer));

  return static_cast<float*>(buffer);
}

template <>
int64_t* DecompressionState::DecompressToBuffer(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  DecompressionStateXtensa dsx(*this);

  dsx.DecompressToBufferWidthAnyInt64_Xtensa(static_cast<int64_t*>(buffer));

  return static_cast<int64_t*>(buffer);
}

#else  // HIFI5

template <typename T>
T* DecompressionState::DecompressToBuffer(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  if (std::is_same<T, int8_t>::value &&
      comp_data_.data.lut_data->compressed_bit_width == 4 &&
      !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth4_16(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 3 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth3_32(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 2 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth2_16(static_cast<int8_t*>(buffer));
  } else {
    DecompressToBufferWidthAny<T>(static_cast<T*>(buffer));
  }

  return static_cast<T*>(buffer);
}

template bool* DecompressionState::DecompressToBuffer<bool>(void*);
template float* DecompressionState::DecompressToBuffer<float>(void*);
template int8_t* DecompressionState::DecompressToBuffer<int8_t>(void*);
template int16_t* DecompressionState::DecompressToBuffer<int16_t>(void*);
template int32_t* DecompressionState::DecompressToBuffer<int32_t>(void*);
template int64_t* DecompressionState::DecompressToBuffer<int64_t>(void*);

#endif  // HIFI5

}  // namespace tflite

#endif  // USE_TFLM_COMPRESSION
