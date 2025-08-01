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

struct TensorInDatum {
  const void* const data;
  const TfLiteIntArray& dims;
};

struct TensorOutDatum {
  void* const data;
  const TfLiteIntArray& dims;
  const TfLiteType type;
  const TfLiteFloatArray& scales;
  const TfLiteIntArray& zero_points;
  const int quantized_dimension;

  // initialized by CreatePerChannelQuantizedTensor
  const TfLiteAffineQuantization affine_quantization;
};

template <typename T, size_t N>
struct AncillaryData {
  AncillaryData() = delete;
  AncillaryData(const uint8_t (&dcm)[tflite::DecodeState::kDcmSizeInBytes],
                const T (&values)[N]) {
    std::copy(std::begin(dcm), std::end(dcm), std::begin(dcm_));
    std::copy(std::begin(values), std::end(values), std::begin(value_table_));
  }

 private:
  uint8_t dcm_[tflite::DecodeState::kDcmSizeInBytes];
  T value_table_[N > 0 ? N : 1];  // assure not zero length
};

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

//
// Prune test data
//
constexpr int8_t kAncillaryDataPrune0[] = {1, 2, 3, 4};
constexpr int16_t kAncillaryDataPrune1[] = {5, 6, 7, 8};
constexpr float kAncillaryDataPrune2[] = {9.0f, 10.0f, 11.0f, 12.0f};
constexpr int8_t kAncillaryDataPrune3[] = {13, 14, 15, 16};
constexpr int8_t kAncillaryDataPrune4[] = {17, 18, 19, 20};

constexpr uint8_t kDcmPrune[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypePrune,  // type: Prune
    1,                                   // DCM version: 1
    0,                                   // reserved
    0,                                   // reserved
    1,                                   // Prune version: 1
};

// Align the tensor data the same as a Buffer in the TfLite schema
alignas(16) const uint8_t kEncodedPrune[] = {0xA5};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapePrune[] = {2, 2, 4};
constexpr int kEncodedShapePrune[] = {1, sizeof(kEncodedPrune)};

// Quantization datum as TfLiteIntArray.
// Scales are modified by FloatArrayFromFloats. As globals they cannot be
// <const> without causing a processor exception.
float kScalesPrune0[] = {2, 1.0f, 1.0f};
constexpr int kZeroPointsPrune0[] = {2, -128, -64};
float kScalesPrune1[] = {4, 1.0f, 1.0f, 1.0f, 1.0f};
constexpr int kZeroPointsPrune1[] = {4, 0, 0, 0, 0};
float kScalesPrune4[] = {4, 1.0f, 1.0f, 1.0f, 1.0f};
constexpr int kZeroPointsPrune4[] = {4, -126, -62, -30, -14};

constexpr int8_t kExpectPrune0[] = {1, -128, 2, -128, -64, 3, -64, 4};
constexpr int16_t kExpectPrune1[] = {5, 0, 6, 0, 0, 7, 0, 8};
constexpr float kExpectPrune2[] = {9.0f, 0.0f,  10.0f, 0.0f,
                                   0.0f, 11.0f, 0.0f,  12.0f};
constexpr int8_t kExpectPrune3[] = {13, 0, 14, 0, 0, 15, 0, 16};
constexpr int8_t kExpectPrune4[] = {17, -62, 18, -14, -126, 19, -30, 20};

template <typename T>
TfLiteStatus CheckOutput(const TfLiteTensor& output,
                         const void* const expected) {
  const T* const expected_data = reinterpret_cast<const T*>(expected);
  const T* const output_data = tflite::GetTensorData<T>(&output);

  constexpr float kTolerance = 1e-5;
  const size_t kOutputCount = tflite::NumElements(&output);
  for (size_t i = 0; i < kOutputCount; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
    TF_LITE_MICRO_CHECK_FAIL();
  }

  return kTfLiteOk;
}

template <size_t kNumInputs, size_t kNumOutputs>
TfLiteStatus ExecuteDecodeTest(
    TfLiteTensor* tensors, const TFLMRegistration& registration,
    const std::initializer_list<const void*>& expected) {
  int kInputArrayData[kNumInputs + 1] = {kNumInputs};
  for (size_t i = 0; i < kNumInputs; i++) {
    kInputArrayData[i + 1] = i;
  }
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);

  int kOutputArrayData[kNumOutputs + 1] = {kNumOutputs};
  for (size_t i = 0; i < kNumOutputs; i++) {
    kOutputArrayData[i + 1] = i + kNumInputs;
  }
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  micro::KernelRunner runner(registration, tensors, kNumInputs + kNumOutputs,
                             inputs_array, outputs_array, nullptr);

  if (runner.InitAndPrepare() != kTfLiteOk || runner.Invoke() != kTfLiteOk) {
    return kTfLiteError;
  }

  const TfLiteTensor* const output_tensors = &tensors[kNumInputs];
  TfLiteStatus status = kTfLiteError;
  for (size_t i = 0; i < kNumOutputs; i++) {
    switch (output_tensors[i].type) {
      case kTfLiteInt8:
        status = CheckOutput<int8_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteInt16:
        status = CheckOutput<int16_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteFloat32:
        status = CheckOutput<float>(output_tensors[i], expected.begin()[i]);
        break;
      default:
        TF_LITE_MICRO_FAIL("unsupported tensor type in test");
        break;
    }
  }

  return status;
}

template <size_t kNumInputs, size_t kNumOutputs>
void TestDecode(const std::initializer_list<const TensorInDatum*>& encodes,
                const std::initializer_list<const TensorInDatum*>& ancillaries,
                const std::initializer_list<const TensorOutDatum*>& outputs,
                const std::initializer_list<const void*>& expected,
                const TFLMRegistration& registration,
                const TfLiteStatus expected_status = kTfLiteOk) {
  TfLiteTensor tensors[kNumInputs + kNumOutputs] = {};

  for (size_t i = 0; i < kNumInputs; i += 2) {
    const TensorInDatum& tid_encode = *encodes.begin()[i / 2];
    tensors[i] = CreateTensor(tid_encode.data,
                              const_cast<TfLiteIntArray*>(&tid_encode.dims),
                              false, kTfLiteUInt8);
    const TensorInDatum& tid_ancillary = *ancillaries.begin()[i / 2];
    tensors[i + 1] = CreateTensor(
        tid_ancillary.data, const_cast<TfLiteIntArray*>(&tid_ancillary.dims),
        false, kTfLiteUInt8);
  }
  for (size_t i = 0; i < kNumOutputs; i++) {
    const TensorOutDatum& tod = *outputs.begin()[i];
    if (tod.scales.size == 0) {
      tensors[i + kNumInputs] = CreateTensor(
          tod.data, const_cast<TfLiteIntArray*>(&tod.dims), false, tod.type);
    } else {
      tensors[i + kNumInputs] = CreatePerChannelQuantizedTensor(
          tod.data, const_cast<TfLiteIntArray*>(&tod.dims),
          const_cast<TfLiteFloatArray*>(&tod.scales),
          const_cast<TfLiteIntArray*>(&tod.zero_points),
          const_cast<TfLiteAffineQuantization*>(&tod.affine_quantization),
          tod.quantized_dimension, false, tod.type);
    }
  }

  TfLiteStatus s = ExecuteDecodeTest<kNumInputs, kNumOutputs>(
      tensors, registration, expected);
  TF_LITE_MICRO_EXPECT_EQ(s, expected_status);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

using tflite::testing::AncillaryData;
using tflite::testing::kAncillaryDataLUT0;
using tflite::testing::kAncillaryDataLUT1;
using tflite::testing::kAncillaryDataPrune0;
using tflite::testing::kAncillaryDataPrune1;
using tflite::testing::kAncillaryDataPrune2;
using tflite::testing::kAncillaryDataPrune3;
using tflite::testing::kAncillaryDataPrune4;
using tflite::testing::kDcmLUT0;
using tflite::testing::kDcmLUT1;
using tflite::testing::kDcmPrune;
using tflite::testing::kEncodedLUT;
using tflite::testing::kEncodedPrune;
using tflite::testing::kEncodedShapeLUT;
using tflite::testing::kEncodedShapePrune;
using tflite::testing::kExpectLUT0;
using tflite::testing::kExpectLUT1;
using tflite::testing::kExpectPrune0;
using tflite::testing::kExpectPrune1;
using tflite::testing::kExpectPrune2;
using tflite::testing::kExpectPrune3;
using tflite::testing::kExpectPrune4;
using tflite::testing::kOutputShapeLUT;
using tflite::testing::kOutputShapePrune;
using tflite::testing::kScalesPrune0;
using tflite::testing::kScalesPrune1;
using tflite::testing::kScalesPrune4;
using tflite::testing::kZeroPointsPrune0;
using tflite::testing::kZeroPointsPrune1;
using tflite::testing::kZeroPointsPrune4;
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
  constexpr float kOutputScalesData[] = {0};
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kOutputScalesData);
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteFloat32,
      *kOutputScales,
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
  constexpr float kOutputScalesData[] = {0};
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kOutputScalesData);
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      *kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
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
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScalesPrune0);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune0);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      *kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune0};

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
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScalesPrune4);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune4);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      *kOutputScales,
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

TF_LITE_MICRO_TEST(DecodePruneQuantizedAltAxisInt16) {
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
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScalesPrune1);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune1);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt16,
      *kOutputScales,
      *kOutputZeroPoints,
      (kOutputDims->size - 1),
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune1};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TESTS_END
