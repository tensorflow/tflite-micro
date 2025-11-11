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

#include <array>
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/decode_test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

//
// Prune test data
//
constexpr int8_t kAncillaryDataPrune0[] = {
    1,  2,  3,  4,  1,   // chan 0
    2,  3,  4,  1,  2,   // chan 0
    3,  4,  1,  2,  3,   // chan 0
    4,  1,  2,  3,  4,   // chan 0
    11, 12, 13, 14, 11,  // chan 1
    12, 13, 14, 11, 12,  // chan 1
    13, 14, 11, 12, 13,  // chan 1
    14, 11, 12, 13, 14   // chan 1
};
constexpr int16_t kAncillaryDataPrune1[] = {
    5,  6,  7,  8,  5,   // chan 0
    6,  7,  8,  5,  6,   // chan 0
    7,  8,  5,  6,  7,   // chan 0
    8,  5,  6,  7,  8,   // chan 0
    15, 16, 17, 18, 15,  // chan 1
    16, 17, 18, 15, 16,  // chan 1
    17, 18, 15, 16, 17,  // chan 1
    18, 15, 16, 17, 18   // chan 1
};
constexpr float kAncillaryDataPrune2[] = {
    9.0f,  10.0f, 11.0f, 12.0f,  // encoded byte 0
    9.0f,  10.0f, 11.0f, 12.0f,  // encoded byte 1
    9.0f,  10.0f, 11.0f, 12.0f,  // encoded byte 2
    9.0f,  10.0f, 11.0f, 12.0f,  // encoded byte 3
    9.0f,  10.0f, 11.0f, 12.0f,  // encoded byte 4
    19.0f, 20.0f, 21.0f, 22.0f,  // encoded byte 5
    19.0f, 20.0f, 21.0f, 22.0f,  // encoded byte 6
    19.0f, 20.0f, 21.0f, 22.0f,  // encoded byte 7
    19.0f, 20.0f, 21.0f, 22.0f,  // encoded byte 8
    19.0f, 20.0f, 21.0f, 22.0f   // encoded byte 9
};
constexpr int8_t kAncillaryDataPrune3[] = {
    13,  14,  15,  16,  13,   // chan 0
    14,  15,  16,  13,  14,   // chan 0
    15,  16,  13,  14,  15,   // chan 0
    16,  13,  14,  15,  16,   // chan 0
    113, 114, 115, 116, 113,  // chan 1
    114, 115, 116, 113, 114,  // chan 1
    115, 116, 113, 114, 115,  // chan 1
    116, 113, 114, 115, 116   // chan 1
};
constexpr int8_t kAncillaryDataPrune4[] = {
    17, 18, 19, 20, 17, 18, 19, 20, 17, 18,  // group 0
    19, 20, 17, 18, 19, 20, 17, 18, 19, 20,  // group 0
    21, 22, 23, 24, 21, 22, 23, 24, 21, 22,  // group 1
    23, 24, 21, 22, 23, 24, 21, 22, 23, 24,  // group 1
};
constexpr int8_t kAncillaryDataPrune5[] = {
    13, 14, 15, 16, 13,  // chan 0
    14, 15, 16, 13, 14,  // chan 0
    15, 16, 13, 14, 15,  // chan 0
    16, 13, 14, 15, 16,  // chan 0
    23, 24, 25, 26, 23,  // chan 0
    24, 25, 26, 23, 24,  // chan 0
    25, 26, 23, 24, 25,  // chan 0
    26, 23, 24, 25, 26   // chan 0
};

constexpr uint8_t kDcmPrune[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypePrune,  // type: Prune
    1,                                   // DCM version: 1
    0,                                   // reserved
    0,                                   // reserved
    1,                                   // Prune version: 1
};

// Align the tensor data the same as a Buffer in the TfLite schema.
// Use 0x5A in byte 1 to check byte ordering in the low-level code.
alignas(16) const uint8_t kEncodedPrune[] = {0xA5, 0x5A, 0xA5, 0xA5, 0xA5,
                                             0xA5, 0xA5, 0xA5, 0xA5, 0xA5};

// Tensor shapes as TfLiteIntArray
constexpr int kEncodedShapePrune[] = {1, sizeof(kEncodedPrune)};
constexpr int kOutputShapePrune[] = {4, 2, 5, 8, 1};    // 2 channels
constexpr int kOutputShapePrune4[] = {4, 1, 2, 1, 40};  // 40 channels, alt-axis
constexpr int kOutputShapePrune5[] = {4, 1, 8, 10, 1};  // 1 channel

// Quantization datum as TfLiteIntArray.
constexpr int kZeroPointsPrune0[] = {2, -128, 0};
constexpr int kZeroPointsPrune1[] = {2, 0, 0};
constexpr int kZeroPointsPrune1_Invalid[] = {2, 0, -1};
constexpr int kZeroPointsPrune3[] = {2, 0, 0};
constexpr int kZeroPointsPrune4[] = {
    40,  // size
    0,   -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10, -11, -12, -13,
    -14, -15, -16, -17, -18, -19, 0,   -1,  -2,  -3,  -4,  -5,  -6,  -7,
    -8,  -9,  -10, -11, -12, -13, -14, -15, -16, -17, -18, -19,
};
constexpr int kZeroPointsPrune5[] = {1, -44};

constexpr int8_t kExpectPrune0[] = {
    1,    -128, 2,    -128, -128, 3,    -128, 4,     // chan 0
    -128, 1,    -128, 2,    3,    -128, 4,    -128,  // chan 0
    1,    -128, 2,    -128, -128, 3,    -128, 4,     // chan 0
    1,    -128, 2,    -128, -128, 3,    -128, 4,     // chan 0
    1,    -128, 2,    -128, -128, 3,    -128, 4,     // chan 0
    11,   0,    12,   0,    0,    13,   0,    14,    // chan 1
    11,   0,    12,   0,    0,    13,   0,    14,    // chan 1
    11,   0,    12,   0,    0,    13,   0,    14,    // chan 1
    11,   0,    12,   0,    0,    13,   0,    14,    // chan 1
    11,   0,    12,   0,    0,    13,   0,    14     // chan 1
};
constexpr int16_t kExpectPrune1[] = {
    5,  0, 6,  0, 0, 7,  0, 8,   // chan 0
    0,  5, 0,  6, 7, 0,  8, 0,   // chan 0
    5,  0, 6,  0, 0, 7,  0, 8,   // chan 0
    5,  0, 6,  0, 0, 7,  0, 8,   // chan 0
    5,  0, 6,  0, 0, 7,  0, 8,   // chan 0
    15, 0, 16, 0, 0, 17, 0, 18,  // chan 1
    15, 0, 16, 0, 0, 17, 0, 18,  // chan 1
    15, 0, 16, 0, 0, 17, 0, 18,  // chan 1
    15, 0, 16, 0, 0, 17, 0, 18,  // chan 1
    15, 0, 16, 0, 0, 17, 0, 18   // chan 1
};
constexpr float kExpectPrune2[] = {
    9.0f,  0.0f, 10.0f, 0.0f,  0.0f,  11.0f, 0.0f,  12.0f,  // encode byte 0
    0.0f,  9.0f, 0.0f,  10.0f, 11.0f, 0.0f,  12.0f, 0.0f,   // encode byte 1
    9.0f,  0.0f, 10.0f, 0.0f,  0.0f,  11.0f, 0.0f,  12.0f,  // encode byte 2
    9.0f,  0.0f, 10.0f, 0.0f,  0.0f,  11.0f, 0.0f,  12.0f,  // encode byte 3
    9.0f,  0.0f, 10.0f, 0.0f,  0.0f,  11.0f, 0.0f,  12.0f,  // encode byte 4
    19.0f, 0.0f, 20.0f, 0.0f,  0.0f,  21.0f, 0.0f,  22.0f,  // encode byte 5
    19.0f, 0.0f, 20.0f, 0.0f,  0.0f,  21.0f, 0.0f,  22.0f,  // encode byte 6
    19.0f, 0.0f, 20.0f, 0.0f,  0.0f,  21.0f, 0.0f,  22.0f,  // encode byte 7
    19.0f, 0.0f, 20.0f, 0.0f,  0.0f,  21.0f, 0.0f,  22.0f,  // encode byte 8
    19.0f, 0.0f, 20.0f, 0.0f,  0.0f,  21.0f, 0.0f,  22.0f   // encode byte 9
};
constexpr int8_t kExpectPrune3[] = {
    13,  0,  14,  0,  0,  15,  0,  16,   // chan 0
    0,   13, 0,   14, 15, 0,   16, 0,    // chan 0
    13,  0,  14,  0,  0,  15,  0,  16,   // chan 0
    13,  0,  14,  0,  0,  15,  0,  16,   // chan 0
    13,  0,  14,  0,  0,  15,  0,  16,   // chan 0
    113, 0,  114, 0,  0,  115, 0,  116,  // chan 1
    113, 0,  114, 0,  0,  115, 0,  116,  // chan 1
    113, 0,  114, 0,  0,  115, 0,  116,  // chan 1
    113, 0,  114, 0,  0,  115, 0,  116,  // chan 1
    113, 0,  114, 0,  0,  115, 0,  116   // chan 1
};
constexpr int8_t kExpectPrune4[] = {
    17,  -1,  18,  -3,  -4,  19,  -6,  20,  -8,  17,   // group 0
    -10, 18,  19,  -13, 20,  -15, 17,  -17, 18,  -19,  // group 0
    0,   19,  -2,  20,  17,  -5,  18,  -7,  -8,  19,   // group 0
    -10, 20,  17,  -13, 18,  -15, -16, 19,  -18, 20,   // group 0
    21,  -1,  22,  -3,  -4,  23,  -6,  24,  21,  -9,   // group 1
    22,  -11, -12, 23,  -14, 24,  21,  -17, 22,  -19,  // group 1
    0,   23,  -2,  24,  21,  -5,  22,  -7,  -8,  23,   // group 1
    -10, 24,  21,  -13, 22,  -15, -16, 23,  -18, 24    // group 1
};
constexpr int8_t kExpectPrune5[] = {
    13,  -44, 14,  -44, -44, 15,  -44, 16,  -44, 13,   // chan 0
    -44, 14,  15,  -44, 16,  -44, 13,  -44, 14,  -44,  // chan 0
    -44, 15,  -44, 16,  13,  -44, 14,  -44, -44, 15,   // chan 0
    -44, 16,  13,  -44, 14,  -44, -44, 15,  -44, 16,   // chan 0
    23,  -44, 24,  -44, -44, 25,  -44, 26,  23,  -44,  // chan 0
    24,  -44, -44, 25,  -44, 26,  23,  -44, 24,  -44,  // chan 0
    -44, 25,  -44, 26,  23,  -44, 24,  -44, -44, 25,   // chan 0
    -44, 26,  23,  -44, 24,  -44, -44, 25,  -44, 26    // chan 0
};

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

using tflite::testing::AncillaryData;
using tflite::testing::TensorInDatum;
using tflite::testing::TensorOutDatum;

TF_LITE_MICRO_TEST(DecodePruneFloat) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) float output_data[std::size(kExpectPrune2)] = {};
  alignas(16) const AncillaryData<float, std::size(kAncillaryDataPrune2)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune2}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteFloat32,
      kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune2};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune3)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune3)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune3}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data, *kOutputDims, kTfLiteInt8, kOutputScales, *kOutputZeroPoints,
      0,           {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune3};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune3)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune3)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune3}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune3);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data, *kOutputDims, kTfLiteInt8, kOutputScales, *kOutputZeroPoints,
      0,           {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune3};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedMixedZeroPointInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune0)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune0)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune0}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune0);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data, *kOutputDims, kTfLiteInt8, kOutputScales, *kOutputZeroPoints,
      0,           {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune0};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedSingleChannelInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune5)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune5)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune5}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune5);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune5);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data, *kOutputDims, kTfLiteInt8, kOutputScales, *kOutputZeroPoints,
      0,           {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune5};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedAltAxisInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune4)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune4)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune4}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune4);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune4);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      kOutputScales,
      *kOutputZeroPoints,
      (kOutputDims->size - 1),
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune4};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedInt16) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int16_t output_data[std::size(kExpectPrune1)] = {};
  alignas(16) const AncillaryData<int16_t, std::size(kAncillaryDataPrune1)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune1}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune1);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt16,
      kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune1};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedInvalidZeroPointInt16) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int16_t output_data[std::size(kExpectPrune1)] = {};
  alignas(16) const AncillaryData<int16_t, std::size(kAncillaryDataPrune1)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune1}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune1_Invalid);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt16,
      kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune1};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE(),
      nullptr, kTfLiteError);
}

TF_LITE_MICRO_TESTS_END
