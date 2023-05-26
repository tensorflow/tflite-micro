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
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/mock_micro_graph.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestCallOnce(const int subgraph0_invoke_count_golden,
                  const int subgraph1_invoke_count_golden) {
  int inputs_array_data[] = {0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {0};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteCallOnceParams params;
  params.init_subgraph_index = 1;

  const TFLMRegistration registration = tflite::Register_CALL_ONCE();
  micro::KernelRunner runner(registration, nullptr, 0, inputs_array,
                             outputs_array, &params);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  for (int i = 0; i < subgraph0_invoke_count_golden; i++) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  }

  TF_LITE_MICRO_EXPECT_EQ(subgraph1_invoke_count_golden,
                          runner.GetMockGraph()->get_invoke_count(1));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(CallOnceShouldOnlyInvokeSubgraphOnce) {
  tflite::testing::TestCallOnce(1, 1);
  tflite::testing::TestCallOnce(10, 1);
}

TF_LITE_MICRO_TESTS_END
