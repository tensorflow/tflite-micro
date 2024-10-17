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

#include <algorithm>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro//micro_log.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
struct TestingInfo {
  T* output;
  T* goldens;
  uint8_t* compressed;
  T* value_table;

  size_t bit_width;
  size_t total_elements;
  size_t total_value_table_elements;
  size_t channel_count;
  bool use_alt_axis;
};

template <typename T>
struct TestingData7_2_256 {
  static constexpr size_t kBitWidth = 7;
  static constexpr size_t kChannels = 2;
  static constexpr size_t kElementsPerChannel = 256;

  static constexpr size_t kTotalElements = kElementsPerChannel * kChannels;
  static constexpr size_t kCompressedBytes =
      ((kTotalElements * kBitWidth) + 7) / 8;
  static constexpr size_t kValueTableSize = (1 << kBitWidth) * kChannels;

  alignas(MicroArenaBufferAlignment()) T output[kTotalElements];
  alignas(MicroArenaBufferAlignment()) uint8_t compressed[kCompressedBytes];
  alignas(MicroArenaBufferAlignment()) T value_table[kValueTableSize];
  T goldens[kTotalElements];
};

TestingData7_2_256<bool> TestingData7_2_256_Bool;
#ifdef notyet
TestingData7_2_256<float> TestingData7_2_256_Float32;
TestingData7_2_256<int8_t> TestingData7_2_256_Int8;
TestingData7_2_256<int16_t> TestingData7_2_256_Int16;
TestingData7_2_256<int32_t> TestingData7_2_256_Int32;
TestingData7_2_256<int64_t> TestingData7_2_256_Int64;
#endif  // notyet

template <typename T>
void FillValueTable(const size_t total_elements, T* value_table) {
  T fill_value = -1;
  for (size_t i = 0; i < total_elements; i++) {
    value_table[i] = fill_value;
    fill_value -= 1;
  }
}

#ifdef notyet
template <>
void FillValueTable(const size_t total_elements, float* value_table) {
  float fill_value = -1.1f;
  for (size_t i = 0; i < total_elements; i++) {
    value_table[i] = fill_value;
    fill_value -= 1.0f;
  }
}
#endif  // notyet

template <>
void FillValueTable(const size_t total_elements, bool* value_table) {
  bool fill_value = true;
  for (size_t i = 0; i < total_elements; i++) {
    value_table[i] = fill_value;
    fill_value = !fill_value;
  }
}

template <typename T>
void FillGoldens(const size_t total_elements, T* goldens,
                 const size_t value_table_elements, const T* value_table,
                 const size_t channels, const bool use_alt_axis) {
  if (use_alt_axis) {
    const size_t value_table_stride = value_table_elements / channels;
    const size_t element_groups = total_elements / channels;
    size_t value_table_index = 0;  // index within current channel

    for (size_t group = 0; group < element_groups; group++) {
      for (size_t channel = 0; channel < channels; channel++) {
        goldens[(group * channels) + channel] =
            value_table[(channel * value_table_stride) + value_table_index];
      }
      if (++value_table_index == value_table_stride) {
        value_table_index = 0;
      }
    }
  } else {
    const size_t value_table_stride = value_table_elements / channels;
    const size_t elements_per_channel = total_elements / channels;
    size_t value_table_index = 0;  // index within current channel

    for (size_t channel = 0; channel < channels; channel++) {
      for (size_t i = 0; i < elements_per_channel; i++) {
        goldens[(channel * elements_per_channel) + i] =
            value_table[(channel * value_table_stride) + value_table_index++];
        if (value_table_index == value_table_stride) {
          value_table_index = 0;
        }
      }
      value_table_index = 0;
    }
  }
}

// returns index within channel
template <typename T>
size_t FindValueTableIndex(const T value, const T* value_table,
                           const size_t value_table_stride) {
  for (size_t i = 0; i < value_table_stride; i++) {
    if (value == value_table[i]) {
      return i;
    }
  }
  return 0;
}

template <typename T>
void FillCompressed(uint8_t* compressed, const size_t total_golden_elements,
                    const T* goldens, const size_t value_table_stride,
                    const T* value_table, const size_t channels,
                    const bool use_alt_axis, const size_t bit_width) {
  uint16_t bits = 0;
  size_t bits_accumulated = 0;

  if (use_alt_axis) {
    size_t golden_element_groups = total_golden_elements / channels;

    for (size_t group = 0; group < golden_element_groups; group++) {
      for (size_t channel = 0; channel < channels; channel++) {
        size_t value_table_index = FindValueTableIndex(
            goldens[(group * golden_element_groups) + channel],
            &value_table[channel * value_table_stride], value_table_stride);
        bits |= value_table_index << (16 - bits_accumulated - bit_width);
        bits_accumulated += bit_width;
        if (bits_accumulated > 8) {
          *compressed++ = static_cast<uint8_t>(bits >> 8);
          bits <<= 8;
          bits_accumulated -= 8;
        }
      }
    }
  } else {
    size_t golden_elements_per_channel = total_golden_elements / channels;

    for (size_t channel = 0; channel < channels; channel++) {
      for (size_t i = 0; i < golden_elements_per_channel; i++) {
        size_t value_table_index = FindValueTableIndex(
            goldens[(channel * golden_elements_per_channel) + i], value_table,
            value_table_stride);
        bits |= value_table_index << (16 - bits_accumulated - bit_width);
        bits_accumulated += bit_width;
        if (bits_accumulated > 8) {
          *compressed++ = static_cast<uint8_t>(bits >> 8);
          bits <<= 8;
          bits_accumulated -= 8;
        }
      }
      value_table += value_table_stride;
    }
  }
}

template <typename T>
TfLiteStatus TestDecompression(TestingInfo<T>* info) {
  CompressionTensorData ctd = {};
  LookupTableData lut_data = {};
  ctd.scheme = CompressionScheme::kBinQuant;
  ctd.data.lut_data = &lut_data;
  lut_data.compressed_bit_width = info->bit_width;
  lut_data.is_per_channel_quantized = info->channel_count > 1 ? true : false;
  lut_data.use_alternate_axis = info->use_alt_axis;
  lut_data.value_table = info->value_table;
  lut_data.value_table_channel_stride =
      info->total_value_table_elements / info->channel_count;

  DecompressionState ds(info->compressed, info->total_elements, ctd,
                        info->channel_count);

  std::fill_n(info->output, info->total_elements, ~0ULL);
  ds.DecompressToBuffer<T>(info->output);

  for (size_t i = 0; i < info->total_elements; i++) {
    TF_LITE_MICRO_EXPECT_EQ(info->goldens[i], info->output[i]);
    TF_LITE_MICRO_CHECK_FAIL();
  }

  return kTfLiteOk;
}

template <typename T>
void TestBitWidth(size_t bit_width) {
  MicroPrintf("  Testing bit width %d", bit_width);

  TestingInfo<T> info = {};

  if (std::is_same<T, bool>::value) {
    info.output = TestingData7_2_256_Bool.output;
    info.goldens = TestingData7_2_256_Bool.goldens;
    info.compressed = TestingData7_2_256_Bool.compressed;
    info.value_table = TestingData7_2_256_Bool.value_table;
  }

  info.bit_width = bit_width;
  info.channel_count = 1;
  info.total_elements = 16;
  info.total_value_table_elements = 1 << bit_width;
  info.use_alt_axis = false;

  FillValueTable(info.total_value_table_elements, info.value_table);
  FillGoldens(info.total_elements, info.goldens,
              info.total_value_table_elements, info.value_table,
              info.channel_count, info.use_alt_axis);
  FillCompressed(info.compressed, info.total_elements, info.goldens,
                 info.total_value_table_elements / info.channel_count,
                 info.value_table, info.channel_count, info.use_alt_axis,
                 info.bit_width);

  TestDecompression(&info);
}

template <typename T>
void TestAllBitWidths() {
  for (size_t bw = 1; bw <= 7; bw++) {
    TestBitWidth<T>(bw);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestBool) { tflite::testing::TestAllBitWidths<bool>(); }

TF_LITE_MICRO_TESTS_END

#endif  // USE_TFLM_COMPRESSION
