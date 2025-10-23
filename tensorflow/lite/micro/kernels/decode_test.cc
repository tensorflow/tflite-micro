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
#include <initializer_list>
#include <type_traits>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

//
// LUT test data
//
constexpr int kBitWidthLUT = 2;

constexpr int8_t kAncillaryDataLUT0[] = {1, 2, 3, 4};
constexpr int16_t kAncillaryDataLUT1[] = {5, 6, 7, 8};

constexpr uint8_t kDcmLUT0[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeLUT,  // type: LUT
    1,                                 // DCM version: 1
    0,                                 // reserved
    0,                                 // reserved
    1,                                 // LUT version: 1
    kBitWidthLUT,                      // Parameters: bit-width 2
    std::size(kAncillaryDataLUT0),     // channel stride
};

constexpr uint8_t kDcmLUT1[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeLUT,  // type: LUT
    1,                                 // DCM version: 1
    0,                                 // reserved
    0,                                 // reserved
    1,                                 // LUT version: 1
    kBitWidthLUT,                      // Parameters: bit-width 2
    std::size(kAncillaryDataLUT1),     // channel stride
};

// Align the tensor data the same as a Buffer in the TfLite schema
alignas(16) const uint8_t kEncodedLUT[] = {0x1B, 0xE4};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapeLUT[] = {3, 1, 2, 4};
constexpr int kEncodedShapeLUT[] = {1, sizeof(kEncodedLUT)};

constexpr int8_t kExpectLUT0[] = {1, 2, 3, 4, 4, 3, 2, 1};
constexpr int16_t kExpectLUT1[] = {5, 6, 7, 8, 8, 7, 6, 5};

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

using tflite::testing::AncillaryData;
using tflite::testing::kAncillaryDataLUT0;
using tflite::testing::kAncillaryDataLUT1;
using tflite::testing::kDcmLUT0;
using tflite::testing::kDcmLUT1;
using tflite::testing::kEncodedLUT;
using tflite::testing::kEncodedShapeLUT;
using tflite::testing::kExpectLUT0;
using tflite::testing::kExpectLUT1;
using tflite::testing::kOutputShapeLUT;
using tflite::testing::TensorInDatum;
using tflite::testing::TensorOutDatum;

TF_LITE_MICRO_TEST(DecodeSingleTensor) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectLUT0)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataLUT0)>
      kAncillaryData = {{kDcmLUT0}, {kAncillaryDataLUT0}};

  constexpr int kAncillaryShapeLUT[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeLUT);
  static const TensorInDatum tid_encode = {
      kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeLUT);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectLUT0};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodeTwoTensors) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data0[std::size(kExpectLUT0)] = {};
  alignas(16) int16_t output_data1[std::size(kExpectLUT1)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataLUT0)>
      kAncillaryData0 = {{kDcmLUT0}, {kAncillaryDataLUT0}};
  alignas(16) const AncillaryData<int16_t, std::size(kAncillaryDataLUT1)>
      kAncillaryData1 = {{kDcmLUT1}, {kAncillaryDataLUT1}};

  constexpr int kAncillaryShapeLUT0[] = {1, sizeof(kAncillaryData0)};
  constexpr int kAncillaryShapeLUT1[] = {1, sizeof(kAncillaryData1)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeLUT);
  static const TensorInDatum tid_encode0 = {
      kEncodedLUT,
      *encoded_dims,
  };
  static const TensorInDatum tid_encode1 = {
      kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode0, &tid_encode1};

  const TfLiteIntArray* const ancillary_dims0 =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT0);
  static const TensorInDatum tid_ancillary0 = {
      &kAncillaryData0,
      *ancillary_dims0,
  };
  const TfLiteIntArray* const ancillary_dims1 =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT1);
  static const TensorInDatum tid_ancillary1 = {
      &kAncillaryData1,
      *ancillary_dims1,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary0, &tid_ancillary1};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeLUT);
  constexpr float output_scales_data[] = {1, 1.0f};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {1, 0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod0 = {
      output_data0,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static const TensorOutDatum tod1 = {
      output_data1,
      *output_dims,
      kTfLiteInt16,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod0, &tod1};

  const std::initializer_list<const void*> expected = {kExpectLUT0,
                                                       kExpectLUT1};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TESTS_END
