/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Fails on unpatched HEAD (Prepare succeeds) and passes after Setup bounds.
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/decode_test_helpers.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

constexpr int kBitWidthLUT = 2;
constexpr int8_t kAncillaryDataLUT0[] = {1, 2, 3, 4};
constexpr uint8_t kDcmLUT0[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeLUT, 1, 0, 0, 1, kBitWidthLUT,
    std::size(kAncillaryDataLUT0),
};
alignas(16) const uint8_t kEncodedLUT[] = {0x1B, 0xE4};
constexpr int kAttackIndexOutputShape[] = {2, 1, 64};
constexpr int kAttackIndexEncodedShape[] = {1, 2};

constexpr int kBitWidth7 = 7;
constexpr int kStrideChannels = 8;
constexpr int kAttackStrideOutputShape[] = {2, 1, 8};
constexpr int kAttackStrideEncodedShape[] = {1, 7};
alignas(16) const uint8_t kEncodedStride[] = {0x7F, 0x5A, 0x7F, 0x5A, 0x7F,
                                              0x5A, 0x7F};
constexpr uint8_t kDcmStride[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeLUT, 1, 0, 0, 1, kBitWidth7, 255,
};
constexpr int8_t kStrideTable[8] = {10, 20, 30, 40, 50, 60, 70, 80};

TfLiteStatus PrepareOnly(TfLiteTensor* tensors, int n_in, int n_out) {
  int in_data[8] = {n_in};
  for (int i = 0; i < n_in; ++i) in_data[i + 1] = i;
  TfLiteIntArray* inputs = tflite::testing::IntArrayFromInts(in_data);
  int out_data[8] = {n_out};
  for (int i = 0; i < n_out; ++i) out_data[i + 1] = i + n_in;
  TfLiteIntArray* outputs = tflite::testing::IntArrayFromInts(out_data);
  tflite::micro::KernelRunner runner(tflite::Register_DECODE(), tensors,
                                     n_in + n_out, inputs, outputs, nullptr);
  return runner.InitAndPrepare();
}

}  // namespace

using tflite::testing::AncillaryData;

TEST(DecodeLutBounds, UndersizedInputRejected) {
  alignas(16) int8_t output_data[64] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataLUT0)>
      kAncillaryData = {{kDcmLUT0}, {kAncillaryDataLUT0}};
  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* encoded_dims =
      tflite::testing::IntArrayFromInts(kAttackIndexEncodedShape);
  const TfLiteIntArray* ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  const TfLiteIntArray* output_dims =
      tflite::testing::IntArrayFromInts(kAttackIndexOutputShape);

  TfLiteTensor tensors[3] = {};
  tensors[0] = tflite::testing::CreateTensor(
      kEncodedLUT, const_cast<TfLiteIntArray*>(encoded_dims), false,
      kTfLiteUInt8);
  tensors[0].allocation_type = kTfLiteMmapRo;
  tensors[1] = tflite::testing::CreateTensor(
      &kAncillaryData, const_cast<TfLiteIntArray*>(ancillary_dims), false,
      kTfLiteUInt8);
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2] = tflite::testing::CreateTensor(
      output_data, const_cast<TfLiteIntArray*>(output_dims), false,
      kTfLiteInt8);

  ASSERT_EQ(PrepareOnly(tensors, 2, 1), kTfLiteError);
}

TEST(DecodeLutBounds, OversizeStrideRejected) {
  alignas(16) int8_t output_data[8] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kStrideTable)> kAncillaryData =
      {{kDcmStride}, {kStrideTable}};
  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* encoded_dims =
      tflite::testing::IntArrayFromInts(kAttackStrideEncodedShape);
  const TfLiteIntArray* ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  const TfLiteIntArray* output_dims =
      tflite::testing::IntArrayFromInts(kAttackStrideOutputShape);

  static float scales_data[kStrideChannels + 1];
  scales_data[0] = static_cast<float>(kStrideChannels);
  for (int i = 1; i <= kStrideChannels; ++i) scales_data[i] = 1.0f;
  const TfLiteFloatArray* scales =
      tflite::testing::FloatArrayFromFloats(scales_data);
  static int zp_data[kStrideChannels + 1];
  zp_data[0] = kStrideChannels;
  const TfLiteIntArray* zps = tflite::testing::IntArrayFromInts(zp_data);
  TfLiteAffineQuantization aq = {};

  TfLiteTensor tensors[3] = {};
  tensors[0] = tflite::testing::CreateTensor(
      kEncodedStride, const_cast<TfLiteIntArray*>(encoded_dims), false,
      kTfLiteUInt8);
  tensors[0].allocation_type = kTfLiteMmapRo;
  tensors[1] = tflite::testing::CreateTensor(
      &kAncillaryData, const_cast<TfLiteIntArray*>(ancillary_dims), false,
      kTfLiteUInt8);
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2] = tflite::testing::CreatePerChannelQuantizedTensor(
      output_data, const_cast<TfLiteIntArray*>(output_dims),
      const_cast<TfLiteFloatArray*>(scales), const_cast<TfLiteIntArray*>(zps),
      &aq, /*quantized_dimension=*/0, false, kTfLiteInt8);

  ASSERT_EQ(PrepareOnly(tensors, 2, 1), kTfLiteError);
}

TF_LITE_MICRO_TESTS_MAIN
