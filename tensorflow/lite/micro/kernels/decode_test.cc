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
struct AncillaryLUT {
  AncillaryLUT(const uint8_t (&dcm)[tflite::DecodeState::kDcmSizeInBytes],
               const T (&values)[N]) {
    std::copy(std::begin(dcm), std::end(dcm), std::begin(dcm_));
    std::copy(std::begin(values), std::end(values), std::begin(value_table_));
  }

 private:
  uint8_t dcm_[tflite::DecodeState::kDcmSizeInBytes];
  T value_table_[N > 0 ? N : 1];  // assure not zero length
};

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
alignas(16) const
    AncillaryLUT<int8_t, std::size(kAncillaryDataLUT0)> kAncillaryLUT0 = {
        {kDcmLUT0}, {kAncillaryDataLUT0}};
alignas(16) const
    AncillaryLUT<int16_t, std::size(kAncillaryDataLUT1)> kAncillaryLUT1 = {
        {kDcmLUT1}, {kAncillaryDataLUT1}};
alignas(16) const uint8_t kEncodedLUT[] = {0x1B, 0xE4};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapeLUT[] = {3, 1, 2, 4};
constexpr int kEncodedShapeLUT[] = {1, sizeof(kEncodedLUT)};
constexpr int kAncillaryShapeLUT0[] = {1, sizeof(kAncillaryLUT0)};
constexpr int kAncillaryShapeLUT1[] = {1, sizeof(kAncillaryLUT1)};

constexpr int8_t kExpectLUT0[] = {1, 2, 3, 4, 4, 3, 2, 1};
constexpr int16_t kExpectLUT1[] = {5, 6, 7, 8, 8, 7, 6, 5};

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

TF_LITE_MICRO_TEST(DecodeSingleTensor) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(tflite::testing::kExpectLUT0)] = {};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kEncodedShapeLUT);
  static const tflite::testing::TensorInDatum tid_encode = {
      tflite::testing::kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const tflite::testing::TensorInDatum*>
      encodes = {
          &tid_encode,
      };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kAncillaryShapeLUT0);
  static const tflite::testing::TensorInDatum tid_ancillary = {
      &tflite::testing::kAncillaryLUT0,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const tflite::testing::TensorInDatum*>
      ancillaries = {&tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kOutputShapeLUT);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const tflite::testing::TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const tflite::testing::TensorOutDatum*>
      outputs = {&tod};

  const std::initializer_list<const void*> expected = {
      tflite::testing::kExpectLUT0,
  };

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodeTwoTensors) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data0[std::size(tflite::testing::kExpectLUT0)] = {};
  alignas(16)
      int16_t output_data1[std::size(tflite::testing::kExpectLUT1)] = {};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kEncodedShapeLUT);
  static const tflite::testing::TensorInDatum tid_encode0 = {
      tflite::testing::kEncodedLUT,
      *encoded_dims,
  };
  static const tflite::testing::TensorInDatum tid_encode1 = {
      tflite::testing::kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const tflite::testing::TensorInDatum*>
      encodes = {&tid_encode0, &tid_encode1};

  const TfLiteIntArray* const ancillary_dims0 =
      tflite::testing::IntArrayFromInts(tflite::testing::kAncillaryShapeLUT0);
  static const tflite::testing::TensorInDatum tid_ancillary0 = {
      &tflite::testing::kAncillaryLUT0,
      *ancillary_dims0,
  };
  const TfLiteIntArray* const ancillary_dims1 =
      tflite::testing::IntArrayFromInts(tflite::testing::kAncillaryShapeLUT1);
  static const tflite::testing::TensorInDatum tid_ancillary1 = {
      &tflite::testing::kAncillaryLUT1,
      *ancillary_dims1,
  };
  static constexpr std::initializer_list<const tflite::testing::TensorInDatum*>
      ancillaries = {&tid_ancillary0, &tid_ancillary1};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kOutputShapeLUT);
  constexpr float output_scales_data[] = {1, 1.0f};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {1, 0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const tflite::testing::TensorOutDatum tod0 = {
      output_data0,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static const tflite::testing::TensorOutDatum tod1 = {
      output_data1,
      *output_dims,
      kTfLiteInt16,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const tflite::testing::TensorOutDatum*>
      outputs = {&tod0, &tod1};

  const std::initializer_list<const void*> expected = {
      tflite::testing::kExpectLUT0,
      tflite::testing::kExpectLUT1,
  };

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TESTS_END
