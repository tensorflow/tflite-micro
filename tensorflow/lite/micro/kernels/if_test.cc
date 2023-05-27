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
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/mock_micro_graph.h"
#include "tensorflow/lite/micro/test_helper_custom_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestIf(int* input1_dims_data, const bool* input1_data,
            int* input2_dims_data, float* input2_data, int* output_dims_data,
            const float* expected_output_data,
            const int subgraph1_invoke_count_golden,
            const int subgraph2_invoke_count_golden, float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteIfParams params;
  params.then_subgraph_index = 1;
  params.else_subgraph_index = 2;

  const TFLMRegistration registration = tflite::Register_IF();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, &params);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());

  TF_LITE_MICRO_EXPECT_TRUE(runner.ValidateTempBufferDeallocated());

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  TF_LITE_MICRO_EXPECT_EQ(output_dims_count, 2);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }

  TF_LITE_MICRO_EXPECT_EQ(subgraph1_invoke_count_golden,
                          runner.GetMockGraph()->get_invoke_count(1));
  TF_LITE_MICRO_EXPECT_EQ(subgraph2_invoke_count_golden,
                          runner.GetMockGraph()->get_invoke_count(2));

  TF_LITE_MICRO_EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(IfShouldInvokeSubgraphWithMockModelConditionTrue) {
  int shape[] = {2, 1, 2};
  int condition_shape[] = {1, 1};
  const bool condition[] = {true};
  float input[] = {5.0, 2.0};
  const float golden[] = {5.0, 2.0};
  float output_data[2] = {0};
  tflite::testing::TestIf(condition_shape, condition, shape, input, shape,
                          golden, 1, 0, output_data);
}

TF_LITE_MICRO_TEST(IfShouldInvokeSubgraphWithMockModelConditionFalse) {
  int shape[] = {2, 1, 2};
  int condition_shape[] = {1, 1};
  const bool condition[] = {false};
  float input[] = {5.0, 2.0};
  const float golden[] = {5.0, 2.0};
  float output_data[2] = {0};
  tflite::testing::TestIf(condition_shape, condition, shape, input, shape,
                          golden, 0, 1, output_data);
}

TF_LITE_MICRO_TEST(IfShouldInvokeSubgraphConditionTrue) {
  constexpr int kArenaSize = 5000;
  uint8_t arena[kArenaSize];

  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithSubgraphsAndIf();
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddIf();
  resolver.AddAdd();
  resolver.AddMul();
  tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* condition = interpreter.input(0);
  TfLiteTensor* input1 = interpreter.input(1);
  TfLiteTensor* input2 = interpreter.input(2);
  TfLiteTensor* output = interpreter.output(0);
  float input1_data[] = {2.0, 5.0};
  float input2_data[] = {3.0, 7.0};
  memcpy(input1->data.f, input1_data, 2 * sizeof(float));
  memcpy(input2->data.f, input2_data, 2 * sizeof(float));
  condition->data.b[0] = true;

  interpreter.Invoke();

  TF_LITE_MICRO_EXPECT_EQ(output->data.f[0], 5.0f);
  TF_LITE_MICRO_EXPECT_EQ(output->data.f[1], 12.0f);
}

TF_LITE_MICRO_TEST(IfShouldInvokeSubgraphConditionFalse) {
  constexpr int kArenaSize = 5000;
  uint8_t arena[kArenaSize];

  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithSubgraphsAndIf();
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddIf();
  resolver.AddAdd();
  resolver.AddMul();
  tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* condition = interpreter.input(0);
  TfLiteTensor* input1 = interpreter.input(1);
  TfLiteTensor* input2 = interpreter.input(2);
  TfLiteTensor* output = interpreter.output(0);
  float input1_data[] = {2.0, 5.0};
  float input2_data[] = {3.0, 7.0};
  memcpy(input1->data.f, input1_data, 2 * sizeof(float));
  memcpy(input2->data.f, input2_data, 2 * sizeof(float));
  condition->data.b[0] = false;

  interpreter.Invoke();

  TF_LITE_MICRO_EXPECT_EQ(output->data.f[0], 6.0f);
  TF_LITE_MICRO_EXPECT_EQ(output->data.f[1], 35.0f);
}

TF_LITE_MICRO_TEST(IfShouldNotOverwriteTensorAcrossSubgraphs) {
  constexpr int kArenaSize = 5000;
  uint8_t arena[kArenaSize];

  const tflite::Model* model =
      tflite::testing::GetModelWithIfAndSubgraphInputTensorOverlap();

  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, arena, kArenaSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());

  TfLiteTensor* condition = interpreter.input(0);
  TfLiteTensor* input1 = interpreter.input(1);
  TfLiteTensor* input2 = interpreter.input(2);
  TfLiteTensor* output = interpreter.output(0);
  constexpr int32_t block_size =
      tflite::MicroArenaBufferAlignment() / sizeof(int32_t);
  int32_t input1_data[2 * block_size] = {1, 1, 1, 1, 2, 2, 2, 2};
  int32_t input2_data[4 * block_size] = {3, 3, 3, 3, 4, 4, 4, 4,
                                         5, 5, 5, 5, 6, 6, 6, 6};
  memcpy(input1->data.i32, input1_data, 2 * block_size * sizeof(int32_t));
  memcpy(input2->data.i32, input2_data, 4 * block_size * sizeof(int32_t));
  condition->data.b[0] = true;

  interpreter.Invoke();
  // Input1 and input2 are first concatenated, then cut to 3 blocks;
  // the new tensor of size 3 is then concatenated with input2.
  int32_t expect_output_data[8 * block_size] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                                                3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                                                5, 5, 6, 6, 6, 6, 0, 0, 0, 0};
  for (int i = 0; i < 8 * block_size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output->data.i32[i], expect_output_data[i]);
  }
}

TF_LITE_MICRO_TESTS_END
