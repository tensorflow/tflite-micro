/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/memory_helpers.h"

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(MemoryHelpersTest, TestAlignPointerUp) {
  uint8_t* input0 = reinterpret_cast<uint8_t*>(0);

  uint8_t* input0_aligned1 = tflite::AlignPointerUp(input0, 1);
  EXPECT_EQ(input0, input0_aligned1);

  uint8_t* input0_aligned2 = tflite::AlignPointerUp(input0, 2);
  EXPECT_EQ(input0, input0_aligned2);

  uint8_t* input0_aligned3 = tflite::AlignPointerUp(input0, 3);
  EXPECT_EQ(input0, input0_aligned3);

  uint8_t* input0_aligned16 = tflite::AlignPointerUp(input0, 16);
  EXPECT_EQ(input0, input0_aligned16);

  uint8_t* input23 = reinterpret_cast<uint8_t*>(23);

  uint8_t* input23_aligned1 = tflite::AlignPointerUp(input23, 1);
  EXPECT_EQ(input23, input23_aligned1);

  uint8_t* input23_aligned2 = tflite::AlignPointerUp(input23, 2);
  uint8_t* expected23_aligned2 = reinterpret_cast<uint8_t*>(24);
  EXPECT_EQ(expected23_aligned2, input23_aligned2);

  uint8_t* input23_aligned3 = tflite::AlignPointerUp(input23, 3);
  uint8_t* expected23_aligned3 = reinterpret_cast<uint8_t*>(24);
  EXPECT_EQ(expected23_aligned3, input23_aligned3);

  uint8_t* input23_aligned16 = tflite::AlignPointerUp(input23, 16);
  uint8_t* expected23_aligned16 = reinterpret_cast<uint8_t*>(32);
  EXPECT_EQ(expected23_aligned16, input23_aligned16);
}

TEST(MemoryHelpersTest, TestAlignPointerDown) {
  uint8_t* input0 = reinterpret_cast<uint8_t*>(0);

  uint8_t* input0_aligned1 = tflite::AlignPointerDown(input0, 1);
  EXPECT_EQ(input0, input0_aligned1);

  uint8_t* input0_aligned2 = tflite::AlignPointerDown(input0, 2);
  EXPECT_EQ(input0, input0_aligned2);

  uint8_t* input0_aligned3 = tflite::AlignPointerDown(input0, 3);
  EXPECT_EQ(input0, input0_aligned3);

  uint8_t* input0_aligned16 = tflite::AlignPointerDown(input0, 16);
  EXPECT_EQ(input0, input0_aligned16);

  uint8_t* input23 = reinterpret_cast<uint8_t*>(23);

  uint8_t* input23_aligned1 = tflite::AlignPointerDown(input23, 1);
  EXPECT_EQ(input23, input23_aligned1);

  uint8_t* input23_aligned2 = tflite::AlignPointerDown(input23, 2);
  uint8_t* expected23_aligned2 = reinterpret_cast<uint8_t*>(22);
  EXPECT_EQ(expected23_aligned2, input23_aligned2);

  uint8_t* input23_aligned3 = tflite::AlignPointerDown(input23, 3);
  uint8_t* expected23_aligned3 = reinterpret_cast<uint8_t*>(21);
  EXPECT_EQ(expected23_aligned3, input23_aligned3);

  uint8_t* input23_aligned16 = tflite::AlignPointerDown(input23, 16);
  uint8_t* expected23_aligned16 = reinterpret_cast<uint8_t*>(16);
  EXPECT_EQ(expected23_aligned16, input23_aligned16);
}

TEST(MemoryHelpersTest, TestAlignSizeUp) {
  EXPECT_EQ(static_cast<size_t>(1), tflite::AlignSizeUp(1, 1));
  EXPECT_EQ(static_cast<size_t>(2), tflite::AlignSizeUp(1, 2));
  EXPECT_EQ(static_cast<size_t>(3), tflite::AlignSizeUp(1, 3));
  EXPECT_EQ(static_cast<size_t>(16), tflite::AlignSizeUp(1, 16));

  EXPECT_EQ(static_cast<size_t>(23), tflite::AlignSizeUp(23, 1));
  EXPECT_EQ(static_cast<size_t>(24), tflite::AlignSizeUp(23, 2));
  EXPECT_EQ(static_cast<size_t>(24), tflite::AlignSizeUp(23, 3));
  EXPECT_EQ(static_cast<size_t>(32), tflite::AlignSizeUp(23, 16));
}

TEST(MemoryHelpersTest, TestTemplatedAlignSizeUp) {
  // Test structure to test AlignSizeUp.
  struct alignas(32) TestAlignSizeUp {
    // Opaque blob
    float blob_data[4];
  };

  EXPECT_EQ(static_cast<size_t>(32), tflite::AlignSizeUp<TestAlignSizeUp>());
  EXPECT_EQ(static_cast<size_t>(64), tflite::AlignSizeUp<TestAlignSizeUp>(2));
}

TEST(MemoryHelpersTest, TestTypeSizeOf) {
  size_t size;
  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteFloat16, &size));
  EXPECT_EQ(sizeof(int16_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteFloat32, &size));
  EXPECT_EQ(sizeof(float), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteFloat64, &size));
  EXPECT_EQ(sizeof(double), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteInt16, &size));
  EXPECT_EQ(sizeof(int16_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteInt32, &size));
  EXPECT_EQ(sizeof(int32_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteUInt32, &size));
  EXPECT_EQ(sizeof(uint32_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteUInt8, &size));
  EXPECT_EQ(sizeof(uint8_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteInt8, &size));
  EXPECT_EQ(sizeof(int8_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteInt64, &size));
  EXPECT_EQ(sizeof(int64_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteUInt64, &size));
  EXPECT_EQ(sizeof(uint64_t), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteBool, &size));
  EXPECT_EQ(sizeof(bool), size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteComplex64, &size));
  EXPECT_EQ(sizeof(float) * 2, size);

  EXPECT_EQ(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteComplex128, &size));
  EXPECT_EQ(sizeof(double) * 2, size);

  EXPECT_NE(kTfLiteOk, tflite::TfLiteTypeSizeOf(kTfLiteNoType, &size));
}

TEST(MemoryHelpersTest, TestBytesRequiredForTensor) {
  const tflite::Tensor* tensor100 =
      tflite::testing::Create1dFlatbufferTensor(100);
  size_t bytes;
  size_t type_size;
  EXPECT_EQ(kTfLiteOk,
            tflite::BytesRequiredForTensor(*tensor100, &bytes, &type_size));
  EXPECT_EQ(static_cast<size_t>(400), bytes);
  EXPECT_EQ(static_cast<size_t>(4), type_size);

  const tflite::Tensor* tensor200 =
      tflite::testing::Create1dFlatbufferTensor(200);
  EXPECT_EQ(kTfLiteOk,
            tflite::BytesRequiredForTensor(*tensor200, &bytes, &type_size));
  EXPECT_EQ(static_cast<size_t>(800), bytes);
  EXPECT_EQ(static_cast<size_t>(4), type_size);
}
TF_LITE_MICRO_TESTS_MAIN
