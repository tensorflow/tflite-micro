#include <cstdint>

#include "third_party/tflite_micro/tensorflow/lite/micro/all_ops_resolver.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/kernels/kernel_runner.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/test_helpers.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

template <typename T>
void TestSelect(int* input1_dims_data, const bool* input1_data,
                int* input2_dims_data, const T* input2_data,
                int* input3_dims_data, const T* input3_data,
                int* output_dims_data, T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* input3_dims = IntArrayFromInts(input3_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 3;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(input3_data, input3_dims),
                                        CreateTensor(output_data, output_dims)};

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TfLiteRegistration registration = tflite::Register_SELECT_V2();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void ExpectEqual(int* dims, const T* expected_data, const T* output_data) {
  TfLiteIntArray* dims_array = IntArrayFromInts(dims);
  const int element_count = ElementCount(*dims_array);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

template <typename T>
void ExpectNear(int* dims, const T* expected_data, const T* output_data) {
  TfLiteIntArray* dims_array = IntArrayFromInts(dims);
  const int element_count = ElementCount(*dims_array);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], 1e-5);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SelectBool) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {true, false, true, false};
  const bool input2_data[] = {false, false, false, false};
  const bool input3_data[] = {true, true, true, true};
  const bool expected_output[] = {false, true, false, true};

  bool output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectFloat) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {true, false, true, false};
  const float input2_data[] = {0.1, 0.2, 0.3, 0.4};
  const float input3_data[] = {0.5, 0.6, 0.7, 0.8};
  const float expected_output[] = {0.1, 0.6, 0.3, 0.8};

  float output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectNear(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectUInt8) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {false, true, false, false};
  const uint8_t input2_data[] = {1, 2, 3, 4};
  const uint8_t input3_data[] = {5, 6, 7, 8};
  const uint8_t expected_output[] = {5, 2, 7, 8};

  uint8_t output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectInt8) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {false, true, false, false};
  const int8_t input2_data[] = {1, -2, 3, 4};
  const int8_t input3_data[] = {5, 6, 7, -8};
  const int8_t expected_output[] = {5, -2, 7, -8};

  int8_t output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectInt16) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {false, true, false, false};
  const int16_t input2_data[] = {1, 2, 3, 4};
  const int16_t input3_data[] = {5, 6, 7, 8};
  const int16_t expected_output[] = {5, 2, 7, 8};

  int16_t output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectInt32) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {false, true, false, false};
  const int32_t input2_data[] = {1, 2, 3, 4};
  const int32_t input3_data[] = {5, 6, 7, 8};
  const int32_t expected_output[] = {5, 2, 7, 8};

  int32_t output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt32OneDimensionConditionWithSingleValue) {
  int input1_shape[] = {1, 1};
  int input2_shape[] = {5, 1, 2, 2, 2, 1};
  int input3_shape[] = {4, 1, 2, 2, 1};

  const bool input1_data[] = {false};
  const int32_t input2_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int32_t input3_data[] = {9, 10, 11, 12};
  const int32_t expected_output[] = {9, 10, 11, 12, 9, 10, 11, 12};

  int32_t output_data[8];
  tflite::testing::TestSelect(input1_shape, input1_data, input2_shape,
                              input2_data, input3_shape, input3_data,
                              input2_shape, output_data);
  tflite::testing::ExpectEqual(input2_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt32LesserThan4D) {
  int input1_shape[] = {2, 1, 2};
  int inout_shape[] = {3, 1, 2, 2};

  const bool input1_data[] = {false, true};
  const int32_t input2_data[] = {1, 2, 3, 4};
  const int32_t input3_data[] = {5, 6, 7, 8};
  const int32_t expected_output[] = {5, 2, 7, 4};

  int32_t output_data[4];
  tflite::testing::TestSelect(input1_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt32OnFalseValue) {
  int input1_shape[] = {1, 1};
  int inout_shape[] = {3, 1, 2, 2};

  const bool input1_data[] = {false};
  const int32_t input2_data[] = {1, 2, 3, 4};
  const int32_t input3_data[] = {5, 6, 7, 8};
  const int32_t expected_output[] = {5, 6, 7, 8};

  int32_t output_data[4];
  tflite::testing::TestSelect(input1_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt32) {
  int input1_shape[] = {2, 1, 2};
  int inout_shape[] = {3, 1, 2, 2};

  const bool input1_data[] = {false, true};
  const int32_t input2_data[] = {1, 2, 3, 4};
  const int32_t input3_data[] = {5, 6, 7, 7};
  const int32_t expected_output[] = {5, 2, 7, 4};

  int32_t output_data[4];
  tflite::testing::TestSelect(input1_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt32OneDimensionConditionWithTwoValues) {
  int input1_shape[] = {1, 2};
  int input_shape[] = {4, 2, 1, 2, 1};
  int output_shape[] = {4, 2, 1, 2, 2};

  const bool input1_data[] = {false, true};
  const int32_t input2_data[] = {1, 2, 3, 4};
  const int32_t input3_data[] = {5, 6, 7, 8};
  const int32_t expected_output[] = {5, 1, 6, 2, 7, 3, 8, 4};

  int32_t output_data[8];
  tflite::testing::TestSelect(input1_shape, input1_data, input_shape,
                              input2_data, input_shape, input3_data,
                              output_shape, output_data);
  tflite::testing::ExpectEqual(output_shape, expected_output, output_data);
}

TF_LITE_MICRO_TESTS_END
