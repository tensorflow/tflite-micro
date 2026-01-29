/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string.h>

#include "python/tflite_micro/python_ops_resolver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add0_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add0_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add0_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add10_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add10_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add10_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add10_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add11_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add11_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add11_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add11_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add12_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add12_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add12_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add12_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add13_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add13_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add13_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add13_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add14_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add14_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add14_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add14_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add15_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add15_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add15_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add15_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add16_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add16_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add16_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add16_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add1_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add1_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add1_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add1_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add2_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add2_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add2_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add2_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add3_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add3_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add3_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add3_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add4_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add4_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add4_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add4_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add5_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add5_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add5_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add5_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add6_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add6_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add6_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add6_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add7_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add7_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add7_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add7_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add8_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add8_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add8_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add8_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add9_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add9_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add9_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/add/add9_model_data.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

constexpr size_t kTensorArenaSize = 1024 * 100;
uint8_t tensor_arena[kTensorArenaSize];
bool print_log = false;

namespace tflite {
namespace micro {
namespace {

void RunModel(const uint8_t* model, const int16_t* input0,
              const uint32_t input0_size, const int16_t* input1,
              const uint32_t input1_size, const int16_t* golden,
              const uint32_t golden_size, const char* name) {
  InitializeTarget();
  MicroProfiler profiler;
  PythonOpsResolver op_resolver;

  MicroInterpreter interpreter(GetModel(model), op_resolver, tensor_arena,
                               kTensorArenaSize, nullptr, &profiler);
  interpreter.AllocateTensors();
  TfLiteTensor* input_tensor0 = interpreter.input(0);
  EXPECT_EQ(input_tensor0->bytes, input0_size * sizeof(int16_t));
  memcpy(interpreter.input(0)->data.raw, input0, input_tensor0->bytes);
  TfLiteTensor* input_tensor1 = interpreter.input(1);
  EXPECT_EQ(input_tensor1->bytes, input1_size * sizeof(int16_t));
  memcpy(interpreter.input(1)->data.raw, input1, input_tensor1->bytes);
  if (kTfLiteOk != interpreter.Invoke()) {
    EXPECT_TRUE(false);
    return;
  }
  if (print_log) {
    profiler.Log();
  }
  MicroPrintf("");

  TfLiteTensor* output_tensor = interpreter.output(0);
  EXPECT_EQ(output_tensor->bytes, golden_size * sizeof(int16_t));
  int16_t* output = ::tflite::GetTensorData<int16_t>(output_tensor);
  for (uint32_t i = 0; i < golden_size; i++) {
    // TODO(b/205046520): Better understand why TfLite and TFLM can sometimes be
    // off by 1.
    EXPECT_NEAR(golden[i], output[i], 1);
  }
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TEST(IntegrationTests, add0_test) {
  tflite::micro::RunModel(
      g_add0_model_data, g_add0_input0_int16_test_data,
      g_add0_input0_int16_test_data_size, g_add0_input1_int16_test_data,
      g_add0_input1_int16_test_data_size, g_add0_golden_int16_test_data,
      g_add0_golden_int16_test_data_size, "add0 test");
}

TEST(IntegrationTests, add1_test) {
  tflite::micro::RunModel(
      g_add1_model_data, g_add1_input0_int16_test_data,
      g_add1_input0_int16_test_data_size, g_add1_input1_int16_test_data,
      g_add1_input1_int16_test_data_size, g_add1_golden_int16_test_data,
      g_add1_golden_int16_test_data_size, "add1 test");
}

TEST(IntegrationTests, add2_test) {
  tflite::micro::RunModel(
      g_add2_model_data, g_add2_input0_int16_test_data,
      g_add2_input0_int16_test_data_size, g_add2_input1_int16_test_data,
      g_add2_input1_int16_test_data_size, g_add2_golden_int16_test_data,
      g_add2_golden_int16_test_data_size, "add2 test");
}

TEST(IntegrationTests, add3_test) {
  tflite::micro::RunModel(
      g_add3_model_data, g_add3_input0_int16_test_data,
      g_add3_input0_int16_test_data_size, g_add3_input1_int16_test_data,
      g_add3_input1_int16_test_data_size, g_add3_golden_int16_test_data,
      g_add3_golden_int16_test_data_size, "add3 test");
}

TEST(IntegrationTests, add4_test) {
  tflite::micro::RunModel(
      g_add4_model_data, g_add4_input0_int16_test_data,
      g_add4_input0_int16_test_data_size, g_add4_input1_int16_test_data,
      g_add4_input1_int16_test_data_size, g_add4_golden_int16_test_data,
      g_add4_golden_int16_test_data_size, "add4 test");
}

TEST(IntegrationTests, add5_test) {
  tflite::micro::RunModel(
      g_add5_model_data, g_add5_input0_int16_test_data,
      g_add5_input0_int16_test_data_size, g_add5_input1_int16_test_data,
      g_add5_input1_int16_test_data_size, g_add5_golden_int16_test_data,
      g_add5_golden_int16_test_data_size, "add5 test");
}

TEST(IntegrationTests, add6_test) {
  tflite::micro::RunModel(
      g_add6_model_data, g_add6_input0_int16_test_data,
      g_add6_input0_int16_test_data_size, g_add6_input1_int16_test_data,
      g_add6_input1_int16_test_data_size, g_add6_golden_int16_test_data,
      g_add6_golden_int16_test_data_size, "add6 test");
}

TEST(IntegrationTests, add7_test) {
  tflite::micro::RunModel(
      g_add7_model_data, g_add7_input0_int16_test_data,
      g_add7_input0_int16_test_data_size, g_add7_input1_int16_test_data,
      g_add7_input1_int16_test_data_size, g_add7_golden_int16_test_data,
      g_add7_golden_int16_test_data_size, "add7 test");
}

TEST(IntegrationTests, add8_test) {
  tflite::micro::RunModel(
      g_add8_model_data, g_add8_input0_int16_test_data,
      g_add8_input0_int16_test_data_size, g_add8_input1_int16_test_data,
      g_add8_input1_int16_test_data_size, g_add8_golden_int16_test_data,
      g_add8_golden_int16_test_data_size, "add8 test");
}

TEST(IntegrationTests, add9_test) {
  tflite::micro::RunModel(
      g_add9_model_data, g_add9_input0_int16_test_data,
      g_add9_input0_int16_test_data_size, g_add9_input1_int16_test_data,
      g_add9_input1_int16_test_data_size, g_add9_golden_int16_test_data,
      g_add9_golden_int16_test_data_size, "add9 test");
}

TEST(IntegrationTests, add10_test) {
  tflite::micro::RunModel(
      g_add10_model_data, g_add10_input0_int16_test_data,
      g_add10_input0_int16_test_data_size, g_add10_input1_int16_test_data,
      g_add10_input1_int16_test_data_size, g_add10_golden_int16_test_data,
      g_add10_golden_int16_test_data_size, "add10 test");
}

TEST(IntegrationTests, add11_test) {
  tflite::micro::RunModel(
      g_add11_model_data, g_add11_input0_int16_test_data,
      g_add11_input0_int16_test_data_size, g_add11_input1_int16_test_data,
      g_add11_input1_int16_test_data_size, g_add11_golden_int16_test_data,
      g_add11_golden_int16_test_data_size, "add11 test");
}

TEST(IntegrationTests, add12_test) {
  tflite::micro::RunModel(
      g_add12_model_data, g_add12_input0_int16_test_data,
      g_add12_input0_int16_test_data_size, g_add12_input1_int16_test_data,
      g_add12_input1_int16_test_data_size, g_add12_golden_int16_test_data,
      g_add12_golden_int16_test_data_size, "add12 test");
}

TEST(IntegrationTests, add13_test) {
  tflite::micro::RunModel(
      g_add13_model_data, g_add13_input0_int16_test_data,
      g_add13_input0_int16_test_data_size, g_add13_input1_int16_test_data,
      g_add13_input1_int16_test_data_size, g_add13_golden_int16_test_data,
      g_add13_golden_int16_test_data_size, "add13 test");
}

TEST(IntegrationTests, add14_test) {
  tflite::micro::RunModel(
      g_add14_model_data, g_add14_input0_int16_test_data,
      g_add14_input0_int16_test_data_size, g_add14_input1_int16_test_data,
      g_add14_input1_int16_test_data_size, g_add14_golden_int16_test_data,
      g_add14_golden_int16_test_data_size, "add14 test");
}

TEST(IntegrationTests, add15_test) {
  tflite::micro::RunModel(
      g_add15_model_data, g_add15_input0_int16_test_data,
      g_add15_input0_int16_test_data_size, g_add15_input1_int16_test_data,
      g_add15_input1_int16_test_data_size, g_add15_golden_int16_test_data,
      g_add15_golden_int16_test_data_size, "add15 test");
}

TEST(IntegrationTests, add16_test) {
  tflite::micro::RunModel(
      g_add16_model_data, g_add16_input0_int16_test_data,
      g_add16_input0_int16_test_data_size, g_add16_input1_int16_test_data,
      g_add16_input1_int16_test_data_size, g_add16_golden_int16_test_data,
      g_add16_golden_int16_test_data_size, "add16 test");
}

TF_LITE_MICRO_TESTS_MAIN
