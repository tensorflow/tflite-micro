/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void ValidateMirrorPadGoldens(TfLiteTensor* tensors, int tensors_size,
                              const T* golden, T* output, int output_size,
                              TfLiteMirrorPaddingMode mode) {
  TfLiteMirrorPaddingParams builtin_data;
  builtin_data.mode = mode;

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_MIRROR_PAD();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, &builtin_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output[i]);
  }
}

template <typename T>
void TestMirrorPad(int* input_shape, const T* input_data, int* pad_shape,
                   const int32_t* pad_data, int* output_shape,
                   const T* golden_data, TfLiteMirrorPaddingMode mode,
                   T* output_data) {
  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* pad_dims = tflite::testing::IntArrayFromInts(pad_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateTensor(input_data, input_dims),
      tflite::testing::CreateTensor(pad_data, pad_dims),
      tflite::testing::CreateTensor(output_data, output_dims),
  };

  ValidateMirrorPadGoldens(tensors, tensors_size, golden_data, output_data,
                           tflite::ElementCount(*output_dims), mode);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(EmptyPad) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 2, 3};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {0, 0, 0, 0};
  int8_t output_data[6];
  const int8_t golden_data[] = {1, 2, 3, 4, 5, 6};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingReflect, output_data);
}

TF_LITE_MICRO_TEST(PadOneSide_right_Reflect) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 3, 4};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {0, 1, 0, 1};
  int8_t output_data[12];
  const int8_t golden_data[] = {1, 2, 3, 2, 4, 5, 6, 5, 1, 2, 3, 2};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingReflect, output_data);
}

TF_LITE_MICRO_TEST(PadOneSide_left_Reflect) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 3, 4};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {1, 0, 1, 0};
  int8_t output_data[12];
  const int8_t golden_data[] = {5, 4, 5, 6, 2, 1, 2, 3, 5, 4, 5, 6};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingReflect, output_data);
}

TF_LITE_MICRO_TEST(PadOneSide_right_Symmetric) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 3, 4};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {0, 1, 0, 1};
  int8_t output_data[12];
  const int8_t golden_data[] = {1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingSymmetric, output_data);
}

TF_LITE_MICRO_TEST(PadOneSide_left_Symmetric) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 3, 4};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {1, 0, 1, 0};
  int8_t output_data[12];
  const int8_t golden_data[] = {1, 1, 2, 3, 1, 1, 2, 3, 4, 4, 5, 6};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingSymmetric, output_data);
}
TF_LITE_MICRO_TEST(PadBothSides_Symmetric) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 4, 5};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {1, 1, 1, 1};
  int8_t output_data[20];
  const int8_t golden_data[] = {1, 1, 2, 3, 3, 1, 1, 2, 3, 3,
                                4, 4, 5, 6, 6, 4, 4, 5, 6, 6};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingSymmetric, output_data);
}

TF_LITE_MICRO_TEST(PadBothSides_Reflect) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 4, 5};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {1, 1, 1, 1};
  int8_t output_data[20];
  const int8_t golden_data[] = {5, 4, 5, 6, 5, 2, 1, 2, 3, 2,
                                5, 4, 5, 6, 5, 2, 1, 2, 3, 2};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingReflect, output_data);
}

TF_LITE_MICRO_TEST(PadBothSides_Symmetric_Whole) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 6, 9};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {2, 2, 3, 3};
  int8_t output_data[54];
  const int8_t golden_data[] = {6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2,
                                3, 3, 2, 1, 3, 2, 1, 1, 2, 3, 3, 2, 1, 6,
                                5, 4, 4, 5, 6, 6, 5, 4, 6, 5, 4, 4, 5, 6,
                                6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingSymmetric, output_data);
}

TF_LITE_MICRO_TEST(PadBothSides_Reflect_Whole) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 4, 7};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {1, 1, 2, 2};
  int8_t output_data[28];
  const int8_t golden_data[] = {6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1,
                                6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingReflect, output_data);
}

TF_LITE_MICRO_TEST(Pad_Symmetric) {
  int input_shape[] = {2, 2, 3};
  int pad_shape[] = {2, 2, 2};
  int output_shape[] = {2, 4, 7};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {1, 1, 2, 2};
  int8_t output_data[28];
  const int8_t golden_data[] = {2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2,
                                5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingSymmetric, output_data);
}

TF_LITE_MICRO_TEST(Pad_1D_Reflect) {
  int input_shape[] = {1, 3};
  int pad_shape[] = {2, 1, 2};
  int output_shape[] = {1, 5};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {0, 2};
  int8_t output_data[5];
  const int8_t golden_data[] = {1, 2, 3, 2, 1};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingReflect, output_data);
}

TF_LITE_MICRO_TEST(Pad_1D_Symmetric) {
  int input_shape[] = {1, 3};
  int pad_shape[] = {2, 1, 2};
  int output_shape[] = {1, 5};

  const int8_t input_data[] = {1, 2, 3, 4, 5, 6};
  const int32_t pad_data[] = {0, 2};
  int8_t output_data[5];
  const int8_t golden_data[] = {1, 2, 3, 3, 2};

  tflite::testing::TestMirrorPad(input_shape, input_data, pad_shape, pad_data,
                                 output_shape, golden_data,
                                 kTfLiteMirrorPaddingSymmetric, output_data);
}

TF_LITE_MICRO_TESTS_END
