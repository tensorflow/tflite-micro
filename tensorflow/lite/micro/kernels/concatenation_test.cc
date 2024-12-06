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
#include <initializer_list>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void TestConcatenateOneInput(int* input1_dims_data, const T* input1_data,
                             int axis, int* output_dims_data, T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(output_data, output_dims)};

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TFLMRegistration registration = Register_CONCATENATION();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestConcatenateTwoInputs(
    int* input1_dims_data, const T* input1_data, int* input2_dims_data,
    const T* input2_data, int axis, int* output_dims_data, T* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<const T> (*comp_info)[2] = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(output_data, output_dims)};

#ifdef USE_TFLM_COMPRESSION

  TestCompressedList<tensors_size> tcl;
  const CompressedTensorList* comp_list_p = nullptr;

  if (comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(tcl.AddInput((*comp_info)[0], tensors[0], 0),
                            kTfLiteOk);
    TF_LITE_MICRO_EXPECT_EQ(tcl.AddInput((*comp_info)[1], tensors[1], 1),
                            kTfLiteOk);
    comp_list_p = tcl.GetCompressedTensorList();
  }

#endif  // USE_TFLM_COMPRESSION

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TFLMRegistration registration = Register_CONCATENATION();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data)
#ifdef USE_TFLM_COMPRESSION
                                 ,
                             nullptr, comp_list_p
#endif  // USE_TFLM_COMPRESSION
  );

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

void TestConcatenateTwoFloatInputs(
    int* input1_dims_data, const float* input1_data, int* input2_dims_data,
    const float* input2_data, int axis, int* output_dims_data,
    const float* expected_output_data, float* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<const float> (*comp_info)[2] = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  TestConcatenateTwoInputs(input1_dims_data, input1_data, input2_dims_data,
                           input2_data, axis, output_dims_data, output_data
#ifdef USE_TFLM_COMPRESSION
                           ,
                           comp_info
#endif  // USE_TFLM_COMPRESSION
  );

  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

template <typename T>
void TestConcatenateQuantizedTwoInputs(
    int* input1_dims_data, const T* input1_data, int* input2_dims_data,
    const T* input2_data, const float input_scale, const int input_zero_point,
    int axis, int* output_dims_data, const T* expected_output_data,
    const float output_scale, const int output_zero_point, T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input_scale,
                            input_zero_point),
      CreateQuantizedTensor(input2_data, input2_dims, input_scale,
                            input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TFLMRegistration registration = Register_CONCATENATION();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

#ifdef USE_TFLM_COMPRESSION

template <typename T>
void TestConcatenateQuantizedTwoInputsCompressed(
    int* input1_dims_data, const uint8_t* input1_data, int* input2_dims_data,
    const uint8_t* input2_data, const float input_scale,
    const int input_zero_point, int axis, int* output_dims_data,
    const T* expected_output_data, const float output_scale,
    const int output_zero_point, T* output_data,
    const TestCompressionInfo<const T> (&comp_info)[2]) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input_scale,
                            input_zero_point, false, typeToTfLiteType<T>()),
      CreateQuantizedTensor(input2_data, input2_dims, input_scale,
                            input_zero_point, false, typeToTfLiteType<T>()),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

#ifdef USE_TFLM_COMPRESSION

  TestCompressedList<tensors_size> tcl;
  const CompressedTensorList* comp_list_p = nullptr;

  TF_LITE_MICRO_EXPECT_EQ(tcl.AddInput(comp_info[0], tensors[0], 0), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(tcl.AddInput(comp_info[1], tensors[1], 1), kTfLiteOk);
  comp_list_p = tcl.GetCompressedTensorList();

#endif  // USE_TFLM_COMPRESSION

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TFLMRegistration registration = Register_CONCATENATION();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&builtin_data), nullptr, comp_list_p);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(BoolTypeOneInput) {
  int input_shape[] = {3, 2, 1, 2};
  int output_shape[] = {3, 2, 1, 2};
  const bool input_value[] = {true, false, false, true};
  int axis = 1;

  bool output_data[4];
  tflite::testing::TestConcatenateOneInput(input_shape, input_value, axis,
                                           output_shape, output_data);

  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(output_shape);
  const int output_dims_count = tflite::ElementCount(*dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(input_value[i], output_data[i]);
  }
}

TF_LITE_MICRO_TEST(BoolTypeTwoInputs) {
  int input1_shape[] = {3, 2, 1, 2};
  const bool input1_value[] = {false, false, false, false};
  int input2_shape[] = {3, 2, 3, 2};
  const bool input2_value[] = {true, true, true, true, true, true,
                               true, true, true, true, true, true};

  const bool expected_output[] = {false, false, true,  true,  true, true,
                                  true,  true,  false, false, true, true,
                                  true,  true,  true,  true};

  const int axis = 1;
  int output_shape[] = {3, 2, 4, 2};
  bool output_data[16];

  tflite::testing::TestConcatenateTwoInputs(input1_shape, input1_value,
                                            input2_shape, input2_value, axis,
                                            output_shape, output_data);

  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(output_shape);
  const int output_dims_count = tflite::ElementCount(*dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output[i], output_data[i]);
  }
}

TF_LITE_MICRO_TEST(TwoInputsAllAxesCombinations) {
  // Concatenate the same two input tensors along all possible axes.

  int input_shape[] = {2, 2, 3};
  const float input1_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float input2_value[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 0
  int output_shape_axis0[] = {2, 4, 3};
  const float output_value_axis0[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 1
  int output_shape_axis1[] = {2, 2, 6};
  const float output_value_axis1[] = {1.0f, 2.0f, 3.0f, 7.0f,  8.0f,  9.0f,
                                      4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};

  float output_data[12];

  // Axis = 0
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 0,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = -2 (equivalent to axis = 0)
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -2,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = 1
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 1,
      output_shape_axis1, output_value_axis1, output_data);

  // Axis = -1 (equivalent to axis = 1)
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -1,
      output_shape_axis1, output_value_axis1, output_data);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(TwoInputsFloatCompressed) {
  int input_shape[] = {2, 2, 3};
  const float input1_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float input2_value[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // Align the tensor data the same as a Buffer in the schema
  alignas(16) const uint8_t inputs_compressed[] = {0x05, 0x39, 0x40};
  constexpr int kBitWidth = 3;

  // expected output when concatenating on axis 0
  int output_shape_axis0[] = {2, 4, 3};
  const float output_value_axis0[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  float output_data[std::extent<decltype(output_value_axis0)>::value];

  tflite::testing::TestCompressionInfo<const float> comp_info[2] = {};
  comp_info[0].scheme = tflite::CompressionScheme::kBinQuant;
  comp_info[0].value_table = input1_value;
  comp_info[0].value_table_stride = std::extent<decltype(input1_value)>::value;
  comp_info[0].bit_width = kBitWidth;
  comp_info[1].scheme = tflite::CompressionScheme::kBinQuant;
  comp_info[1].value_table = input2_value;
  comp_info[1].value_table_stride = std::extent<decltype(input2_value)>::value;
  comp_info[1].bit_width = kBitWidth;

  // Axis = 0
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, reinterpret_cast<const float*>(inputs_compressed),
      input_shape, reinterpret_cast<const float*>(inputs_compressed),
      /* axis */ 0, output_shape_axis0, output_value_axis0, output_data,
      &comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(TwoInputsQuantizedInt8) {
  const int axis = 2;
  int input_shape[] = {3, 2, 1, 2};
  int output_shape[] = {3, 2, 1, 4};

  const float input_scale = 0.1f;
  const int input_zero_point = 0;
  const float output_scale = 0.1f;
  const int output_zero_point = 0;

  const int8_t input1_values[] = {1, 2, 3, 4};

  const int8_t input2_values[] = {5, 6, 7, 8};

  const int8_t output_value[] = {1, 2, 5, 6, 3, 4, 7, 8};

  int8_t output_data[8];
  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input_shape, input2_values, input_scale,
      input_zero_point, axis, output_shape, output_value, output_scale,
      output_zero_point, output_data);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(TwoInputsQuantizedInt8Compressed) {
  const int axis = 2;
  int input_shape[] = {3, 2, 1, 2};
  int output_shape[] = {3, 2, 1, 4};

  const float input_scale = 0.1f;
  const int input_zero_point = 0;
  const float output_scale = 0.1f;
  const int output_zero_point = 0;

  const int8_t input1_values[] = {1, 2, 3, 4};
  const int8_t input2_values[] = {5, 6, 7, 8};
  const int8_t output_value[] = {1, 2, 5, 6, 3, 4, 7, 8};
  // Align the tensor data the same as a Buffer in the schema
  alignas(16) const uint8_t input_compressed[] = {0x1B};
  constexpr int kBitWidth = 2;

  int8_t output_data[std::extent<decltype(output_value)>::value];

  tflite::testing::TestCompressionInfo<const int8_t> comp_info[2] = {};
  comp_info[0].scheme = tflite::CompressionScheme::kBinQuant;
  comp_info[0].value_table = input1_values;
  comp_info[0].value_table_stride = std::extent<decltype(input1_values)>::value;
  comp_info[0].bit_width = kBitWidth;
  comp_info[1].scheme = tflite::CompressionScheme::kBinQuant;
  comp_info[1].value_table = input2_values;
  comp_info[1].value_table_stride = std::extent<decltype(input2_values)>::value;
  comp_info[1].bit_width = kBitWidth;

  tflite::testing::TestConcatenateQuantizedTwoInputsCompressed(
      input_shape, input_compressed, input_shape, input_compressed, input_scale,
      input_zero_point, axis, output_shape, output_value, output_scale,
      output_zero_point, output_data, comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(TwoInputsQuantizedInt16) {
  const int axis = 2;
  int input_shape[] = {3, 2, 1, 2};
  int output_shape[] = {3, 2, 1, 4};

  const float input_scale = 0.1f;
  const int input_zero_point = 0;
  const float output_scale = 0.1f;
  const int output_zero_point = 0;

  const int16_t input1_values[] = {1, 2, 3, 4};

  const int16_t input2_values[] = {5, 6, 7, 8};

  const int16_t output_value[] = {1, 2, 5, 6, 3, 4, 7, 8};

  int16_t output_data[8];
  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input_shape, input2_values, input_scale,
      input_zero_point, axis, output_shape, output_value, output_scale,
      output_zero_point, output_data);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(TwoInputsQuantizedInt16Compressed) {
  const int axis = 2;
  int input_shape[] = {3, 2, 1, 2};
  int output_shape[] = {3, 2, 1, 4};

  const float input_scale = 0.1f;
  const int input_zero_point = 0;
  const float output_scale = 0.1f;
  const int output_zero_point = 0;

  const int16_t input1_values[] = {1, 2, 3, 4};
  const int16_t input2_values[] = {5, 6, 7, 8};
  const int16_t output_value[] = {1, 2, 5, 6, 3, 4, 7, 8};
  // Align the tensor data the same as a Buffer in the schema
  alignas(16) const uint8_t input_compressed[] = {0x1B};
  constexpr int kBitWidth = 2;

  int16_t output_data[std::extent<decltype(output_value)>::value];

  tflite::testing::TestCompressionInfo<const int16_t> comp_info[2] = {};
  comp_info[0].scheme = tflite::CompressionScheme::kBinQuant;
  comp_info[0].value_table = input1_values;
  comp_info[0].value_table_stride = std::extent<decltype(input1_values)>::value;
  comp_info[0].bit_width = kBitWidth;
  comp_info[1].scheme = tflite::CompressionScheme::kBinQuant;
  comp_info[1].value_table = input2_values;
  comp_info[1].value_table_stride = std::extent<decltype(input2_values)>::value;
  comp_info[1].bit_width = kBitWidth;

  tflite::testing::TestConcatenateQuantizedTwoInputsCompressed(
      input_shape, input_compressed, input_shape, input_compressed, input_scale,
      input_zero_point, axis, output_shape, output_value, output_scale,
      output_zero_point, output_data, comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(ThreeDimensionalTwoInputsDifferentShapes) {
  const int axis = 1;

  int input1_shape[] = {3, 2, 1, 2};
  int input2_shape[] = {3, 2, 3, 2};
  int output_shape[] = {3, 2, 4, 2};

  const float input1_values[] = {1.0f, 3.0f, 4.0f, 7.0f};
  const float input2_values[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const float output_values[] = {1.0f, 3.0f,  1.0f,  2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f,  4.0f,  7.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f};

  float output_data[16];
  tflite::testing::TestConcatenateTwoFloatInputs(
      input1_shape, input1_values, input2_shape, input2_values, axis,
      output_shape, output_values, output_data);
}

TF_LITE_MICRO_TEST(TwoInputsFiveDimensionsAllAxesCombinations) {
  // Concatenate the same two input tensors along all possible axes.
  int input_shape[] = {5, 2, 1, 2, 1, 3};
  const int kInputSize = 12;
  const float input1_value[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const float input2_value[] = {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};

  float output_data[2 * kInputSize];

  // Axis = 0
  int output_shape_axis0[] = {5, 4, 1, 2, 1, 3};
  const float output_value_axis0[2 * kInputSize] = {
      1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
      9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
      17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 0,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = 4
  int output_shape_axis4[] = {5, 2, 1, 2, 1, 6};
  const float output_value_axis4[2 * kInputSize] = {
      1.0f,  2.0f,  3.0f,  13.0f, 14.0f, 15.0f, 4.0f,  5.0f,
      6.0f,  16.0f, 17.0f, 18.0f, 7.0f,  8.0f,  9.0f,  19.0f,
      20.0f, 21.0f, 10.0f, 11.0f, 12.0f, 22.0f, 23.0f, 24.0f};
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 4,
      output_shape_axis4, output_value_axis4, output_data);

  // Axis = -2
  int output_shape_axis_minus2[] = {5, 2, 1, 2, 2, 3};
  const float output_value_axis_minus2[2 * kInputSize] = {
      1.0f,  2.0f,  3.0f,  13.0f, 14.0f, 15.0f, 4.0f,  5.0f,
      6.0f,  16.0f, 17.0f, 18.0f, 7.0f,  8.0f,  9.0f,  19.0f,
      20.0f, 21.0f, 10.0f, 11.0f, 12.0f, 22.0f, 23.0f, 24.0f};
  tflite::testing::TestConcatenateTwoFloatInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -2,
      output_shape_axis_minus2, output_value_axis_minus2, output_data);
}

TF_LITE_MICRO_TEST(TwoInputsQuantizedInt8FiveDimensions) {
  const int axis = 2;
  int input_shape[] = {5, 2, 1, 2, 1, 3};
  const int kInputSize = 12;
  const int8_t input1_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const int8_t input2_values[] = {13, 14, 15, 16, 17, 18,
                                  19, 20, 21, 22, 23, 24};

  const float input_scale = 0.1f;
  const int input_zero_point = 0;
  const float output_scale = 0.1f;
  const int output_zero_point = 0;

  const int8_t output_value[] = {1, 2, 3, 4,  5,  6,  13, 14, 15, 16, 17, 18,
                                 7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24};
  int output_shape[] = {5, 2, 1, 4, 1, 3};
  int8_t output_data[2 * kInputSize];

  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input_shape, input2_values, input_scale,
      input_zero_point, axis, output_shape, output_value, output_scale,
      output_zero_point, output_data);
}

TF_LITE_MICRO_TESTS_END
