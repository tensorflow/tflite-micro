/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Input data expects a 4-D tensor of [batch, height, width, channels]
// Output data should match input datas batch and channels
// Expected sizes should be a 1-D tensor with 2 elements: new_height & new_width
template <typename T>
void TestResizeNearestNeighbor(int* input_dims_data, const T* input_data,
                               const int32_t* expected_size_data,
                               const T* expected_output_data,
                               int* output_dims_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);

  int expected_size_dims_data[] = {1, 2};
  TfLiteIntArray* expected_size_dims =
      IntArrayFromInts(expected_size_dims_data);

  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  int output_dims_count = ElementCount(*output_dims);

  constexpr int tensors_size = 3;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(expected_size_data, expected_size_dims),
      CreateTensor(output_data, output_dims),
  };

  tensors[1].allocation_type = kTfLiteMmapRo;

  TfLiteResizeNearestNeighborParams builtin_data = {false, false};

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RESIZE_NEAREST_NEIGHBOR();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, &builtin_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // compare results
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(HorizontalResize) {
  int input_dims[] = {4, 1, 1, 2, 1};
  const float input_data[] = {3, 6};
  const int32_t expected_size_data[] = {1, 3};
  const float expected_output_data[] = {3, 3, 6};
  int output_dims[] = {4, 1, 1, 3, 1};
  float output_data[3];

  tflite::testing::TestResizeNearestNeighbor<float>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(HorizontalResizeInt8) {
  int input_dims[] = {4, 1, 1, 2, 1};
  const int8_t input_data[] = {-3, 6};
  const int32_t expected_size_data[] = {1, 3};
  const int8_t expected_output_data[] = {-3, -3, 6};
  int output_dims[] = {4, 1, 1, 3, 1};
  int8_t output_data[3];

  tflite::testing::TestResizeNearestNeighbor<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(HorizontalResizeInt16) {
  int input_dims[] = {4, 1, 1, 2, 1};
  const int16_t input_data[] = {-3, 6};
  const int32_t expected_size_data[] = {1, 3};
  const int16_t expected_output_data[] = {-3, -3, 6};
  int output_dims[] = {4, 1, 1, 3, 1};
  int16_t output_data[3];

  tflite::testing::TestResizeNearestNeighbor<int16_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(VerticalResize) {
  int input_dims[] = {4, 1, 2, 1, 1};
  const float input_data[] = {3, 9};
  const int32_t expected_size_data[] = {3, 1};
  const float expected_output_data[] = {3, 3, 9};
  int output_dims[] = {4, 1, 3, 1, 1};
  float output_data[3];

  tflite::testing::TestResizeNearestNeighbor<float>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(VerticalResizeInt8) {
  int input_dims[] = {4, 1, 2, 1, 1};
  const int8_t input_data[] = {3, -9};
  const int32_t expected_size_data[] = {3, 1};
  const int8_t expected_output_data[] = {3, 3, -9};
  int output_dims[] = {4, 1, 3, 1, 1};
  int8_t output_data[3];

  tflite::testing::TestResizeNearestNeighbor<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(VerticalResizeInt16) {
  int input_dims[] = {4, 1, 2, 1, 1};
  const int16_t input_data[] = {3, -9};
  const int32_t expected_size_data[] = {3, 1};
  const int16_t expected_output_data[] = {3, 3, -9};
  int output_dims[] = {4, 1, 3, 1, 1};
  int16_t output_data[3];

  tflite::testing::TestResizeNearestNeighbor<int16_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(TwoDimensionalResize) {
  int input_dims[] = {4, 1, 2, 2, 1};
  const float input_data[] = {
      3,
      6,  //
      9,
      12,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const float expected_output_data[] = {
      3, 3, 6,  //
      3, 3, 6,  //
      9, 9, 12  //
  };

  int output_dims[] = {4, 1, 3, 3, 1};
  float output_data[9];

  tflite::testing::TestResizeNearestNeighbor<float>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeInt8) {
  int input_dims[] = {4, 1, 2, 2, 1};
  const int8_t input_data[] = {
      3,
      -6,  //
      9,
      12,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int8_t expected_output_data[] = {
      3, 3, -6,  //
      3, 3, -6,  //
      9, 9, 12,  //
  };
  int output_dims[] = {4, 1, 3, 3, 1};
  int8_t output_data[9];

  tflite::testing::TestResizeNearestNeighbor<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeInt16) {
  int input_dims[] = {4, 1, 2, 2, 1};
  const int16_t input_data[] = {
      3,
      -6,  //
      9,
      12,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int16_t expected_output_data[] = {
      3, 3, -6,  //
      3, 3, -6,  //
      9, 9, 12,  //
  };
  int output_dims[] = {4, 1, 3, 3, 1};
  int16_t output_data[9];

  tflite::testing::TestResizeNearestNeighbor<int16_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatches) {
  int input_dims[] = {4, 2, 2, 2, 1};
  const float input_data[] = {
      3,  6,   //
      9,  12,  //
      4,  10,  //
      10, 16   //
  };
  const int32_t expected_size_data[] = {3, 3};
  const float expected_output_data[] = {
      3,  3,  6,   //
      3,  3,  6,   //
      9,  9,  12,  //
      4,  4,  10,  //
      4,  4,  10,  //
      10, 10, 16,  //
  };
  int output_dims[] = {4, 2, 3, 3, 1};
  float output_data[18];

  tflite::testing::TestResizeNearestNeighbor<float>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatchesInt8) {
  int input_dims[] = {4, 2, 2, 2, 1};
  const int8_t input_data[] = {
      3,  6,    //
      9,  -12,  //
      -4, 10,   //
      10, 16    //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int8_t expected_output_data[] = {
      3,  3,  6,    //
      3,  3,  6,    //
      9,  9,  -12,  //
      -4, -4, 10,   //
      -4, -4, 10,   //
      10, 10, 16,   //
  };
  int output_dims[] = {4, 2, 3, 3, 1};
  int8_t output_data[18];

  tflite::testing::TestResizeNearestNeighbor<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatchesInt16) {
  int input_dims[] = {4, 2, 2, 2, 1};
  const int16_t input_data[] = {
      3,  6,    //
      9,  -12,  //
      -4, 10,   //
      10, 16    //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int16_t expected_output_data[] = {
      3,  3,  6,    //
      3,  3,  6,    //
      9,  9,  -12,  //
      -4, -4, 10,   //
      -4, -4, 10,   //
      10, 10, 16,   //
  };
  int output_dims[] = {4, 2, 3, 3, 1};
  int16_t output_data[18];

  tflite::testing::TestResizeNearestNeighbor<int16_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(ThreeDimensionalResize) {
  int input_dims[] = {4, 1, 2, 2, 2};
  const float input_data[] = {
      3, 4,  6,  10,  //
      9, 10, 12, 16,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const float expected_output_data[] = {
      3, 4,  3, 4,  6,  10,  //
      3, 4,  3, 4,  6,  10,  //
      9, 10, 9, 10, 12, 16,  //
  };
  int output_dims[] = {4, 1, 3, 3, 2};
  float output_data[18];

  tflite::testing::TestResizeNearestNeighbor<float>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(ThreeDimensionalResizeInt8) {
  int input_dims[] = {4, 1, 2, 2, 2};
  const int8_t input_data[] = {
      3,  4,  -6,  10,  //
      10, 12, -14, 16,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int8_t expected_output_data[] = {
      3,  4,  3,  4,  -6,  10,  //
      3,  4,  3,  4,  -6,  10,  //
      10, 12, 10, 12, -14, 16,  //
  };
  int output_dims[] = {4, 1, 3, 3, 2};
  int8_t output_data[18];

  tflite::testing::TestResizeNearestNeighbor<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TEST(ThreeDimensionalResizeInt16) {
  int input_dims[] = {4, 1, 2, 2, 2};
  const int16_t input_data[] = {
      3,  4,  -6,  10,  //
      10, 12, -14, 16,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int16_t expected_output_data[] = {
      3,  4,  3,  4,  -6,  10,  //
      3,  4,  3,  4,  -6,  10,  //
      10, 12, 10, 12, -14, 16,  //
  };
  int output_dims[] = {4, 1, 3, 3, 2};
  int16_t output_data[18];

  tflite::testing::TestResizeNearestNeighbor<int16_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data);
}

TF_LITE_MICRO_TESTS_END
