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
#include <initializer_list>
#include <type_traits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro//micro_log.h"
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
struct TestingData {
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

TestingData<bool> TestingData_Bool;
TestingData<float> TestingData_Float32;
TestingData<int8_t> TestingData_Int8;
TestingData<int16_t> TestingData_Int16;
TestingData<int32_t> TestingData_Int32;
TestingData<int64_t> TestingData_Int64;

template <typename T>
TestingData<T>* GetTestingData();

template <>
TestingData<bool>* GetTestingData() {
  return &TestingData_Bool;
}

template <>
TestingData<float>* GetTestingData() {
  return &TestingData_Float32;
}

template <>
TestingData<int8_t>* GetTestingData() {
  return &TestingData_Int8;
}

template <>
TestingData<int16_t>* GetTestingData() {
  return &TestingData_Int16;
}

template <>
TestingData<int32_t>* GetTestingData() {
  return &TestingData_Int32;
}

template <>
TestingData<int64_t>* GetTestingData() {
  return &TestingData_Int64;
}

template <typename T>
void FillValueTable(const size_t total_elements, T* value_table) {
  T fill_value = -1;
  for (size_t i = 0; i < total_elements; i++) {
    value_table[i] = fill_value;
    fill_value -= 1;
  }
}

template <>
void FillValueTable(const size_t total_elements, float* value_table) {
  float fill_value = -1.1f;
  for (size_t i = 0; i < total_elements; i++) {
    value_table[i] = fill_value;
    fill_value -= 1.0f;
  }
}

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
            value_table[(channel * value_table_stride) + value_table_index];
        if (++value_table_index == value_table_stride) {
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
            goldens[(group * channels) + channel],
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

  if (bits_accumulated > 0) {
    *compressed = static_cast<uint8_t>(bits >> 8);
  }
}

template <typename T>
void GenerateData(TestingInfo<T>& info) {
  FillValueTable(info.total_value_table_elements, info.value_table);
  FillGoldens(info.total_elements, info.goldens,
              info.total_value_table_elements, info.value_table,
              info.channel_count, info.use_alt_axis);
  FillCompressed(info.compressed, info.total_elements, info.goldens,
                 info.total_value_table_elements / info.channel_count,
                 info.value_table, info.channel_count, info.use_alt_axis,
                 info.bit_width);
}

template <typename T>
void TestDataSetup(TestingInfo<T>* info, TestingData<T>* data) {
  info->output = data->output;
  info->goldens = data->goldens;
  info->compressed = data->compressed;
  info->value_table = data->value_table;
}

template <typename T>
TfLiteStatus TestDecompression(TestingInfo<T>* info) {
  GenerateData(*info);

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

  std::fill_n(info->output, info->total_elements, static_cast<T>(~0ULL));
  ds.DecompressToBuffer<T>(info->output);

  bool saved_fail_state = micro_test::did_test_fail;
  micro_test::did_test_fail = false;
  for (size_t i = 0; i < info->total_elements; i++) {
    TF_LITE_MICRO_EXPECT_EQ(info->goldens[i], info->output[i]);
    if (micro_test::did_test_fail) {
      return kTfLiteError;
    }
  }
  micro_test::did_test_fail = saved_fail_state;
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TestValueTable2n(TestingInfo<T>& info) {
  if (std::is_same<T, bool>::value) {
    info.total_value_table_elements = 2 * info.channel_count;
  } else {
    info.total_value_table_elements =
        (1 << info.bit_width) * info.channel_count;
  }
  info.total_value_table_elements =
      std::min(info.total_value_table_elements, info.total_elements);
  info.total_value_table_elements = std::min(info.total_value_table_elements,
                                             TestingData<T>::kValueTableSize);

  MicroPrintf("        Testing value table 2^n: %d",
              info.total_value_table_elements);
  return TestDecompression(&info);
}

template <typename T>
TfLiteStatus TestValueTable2nMinus1(TestingInfo<T>& info) {
  if (std::is_same<T, bool>::value) {
    info.total_value_table_elements = 1 * info.channel_count;
  } else {
    info.total_value_table_elements =
        ((1 << info.bit_width) - 1) * info.channel_count;
  }
  info.total_value_table_elements =
      std::min(info.total_value_table_elements, info.total_elements);
  info.total_value_table_elements = std::min(info.total_value_table_elements,
                                             TestingData<T>::kValueTableSize);

  MicroPrintf("        Testing value table 2^n-1: %d",
              info.total_value_table_elements);
  return TestDecompression(&info);
}

template <typename T>
void TestElementCount(TestingInfo<T>& info) {
  static constexpr std::initializer_list<size_t> elements_per_channel{
      1,   2,
      3,   4,
      5,   7,
      8,   9,
      15,  16,
      17,  31,
      32,  33,
      63,  64,
      65,  127,
      128, 129,
      255, TestingData<T>::kElementsPerChannel};

  MicroPrintf("      Testing element count: %d thru %d",
              elements_per_channel.begin()[0], elements_per_channel.end()[-1]);

  for (size_t i = 0; i < elements_per_channel.size(); i++) {
    info.total_elements = elements_per_channel.begin()[i] * info.channel_count;

    TfLiteStatus s;
    s = TestValueTable2n(info);
    if (s == kTfLiteError) {
      MicroPrintf("       Failed element count: %d", info.total_elements);
    }
    s = TestValueTable2nMinus1(info);
    if (s == kTfLiteError) {
      MicroPrintf("       Failed element count: %d", info.total_elements);
    }
  }
}

template <typename T>
void TestSingleChannel(TestingInfo<T>& info) {
  info.channel_count = 1;

  MicroPrintf("    Testing single channel");
  TestElementCount(info);
}

template <typename T>
void TestMultiChannel(TestingInfo<T>& info) {
  info.channel_count = TestingData<T>::kChannels;

  MicroPrintf("    Testing multiple channels: %d", info.channel_count);
  TestElementCount(info);
}

template <typename T>
void TestBitWidth(TestingInfo<T>& info) {
  info.use_alt_axis = false;

  MicroPrintf("  Testing bit width %d", info.bit_width);
  TestSingleChannel(info);
  TestMultiChannel(info);
}

template <typename T>
void TestBitWidthAltAxis(TestingInfo<T>& info) {
  info.use_alt_axis = true;

  MicroPrintf("  Testing alt-axis bit width %d", info.bit_width);
  TestSingleChannel(info);
  TestMultiChannel(info);
}

template <typename T>
void TestAllBitWidths() {
  TestingInfo<T> info = {};
  TestDataSetup<T>(&info, GetTestingData<T>());

  for (size_t bw = 1; bw <= TestingData<T>::kBitWidth; bw++) {
    info.bit_width = bw;

    TestBitWidth<T>(info);
    TestBitWidthAltAxis<T>(info);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestBool) { tflite::testing::TestAllBitWidths<bool>(); }
TF_LITE_MICRO_TEST(TestFloat) { tflite::testing::TestAllBitWidths<float>(); }
TF_LITE_MICRO_TEST(TestInt8) { tflite::testing::TestAllBitWidths<int8_t>(); }
TF_LITE_MICRO_TEST(TestInt16) { tflite::testing::TestAllBitWidths<int16_t>(); }
TF_LITE_MICRO_TEST(TestInt32) { tflite::testing::TestAllBitWidths<int32_t>(); }
TF_LITE_MICRO_TEST(TestInt64) { tflite::testing::TestAllBitWidths<int64_t>(); }

TF_LITE_MICRO_TESTS_END

#endif  // USE_TFLM_COMPRESSION
