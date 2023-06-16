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
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu0_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu0_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu10_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu10_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu10_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu11_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu11_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu11_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu12_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu12_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu12_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu13_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu13_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu13_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu14_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu14_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu14_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu15_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu15_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu15_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu16_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu16_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu16_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu17_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu17_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu17_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu18_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu18_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu18_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu19_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu19_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu19_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu1_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu1_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu1_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu20_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu20_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu20_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu21_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu21_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu21_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu22_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu22_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu22_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu2_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu2_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu2_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu3_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu3_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu3_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu4_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu4_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu4_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu5_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu5_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu5_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu6_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu6_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu6_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu7_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu7_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu7_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu8_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu8_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu8_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu9_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu9_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/leaky_relu/leaky_relu9_model_data.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

constexpr size_t kTensorArenaSize = 1024 * 100;
uint8_t tensor_arena[kTensorArenaSize];
bool print_log = false;

namespace tflite {
namespace micro {
namespace {

void RunModel(const uint8_t* model, const int16_t* input0,
              const uint32_t input0_size, const int16_t* golden,
              const uint32_t golden_size, const char* name) {
  InitializeTarget();
  MicroProfiler profiler;
  PythonOpsResolver op_resolver;

  MicroInterpreter interpreter(GetModel(model), op_resolver, tensor_arena,
                               kTensorArenaSize, nullptr, &profiler);
  interpreter.AllocateTensors();
  TfLiteTensor* input_tensor0 = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_EQ(input_tensor0->bytes, input0_size * sizeof(int16_t));
  memcpy(interpreter.input(0)->data.raw, input0, input_tensor0->bytes);
  if (kTfLiteOk != interpreter.Invoke()) {
    TF_LITE_MICRO_EXPECT(false);
    return;
  }
  if (print_log) {
    profiler.Log();
  }
  MicroPrintf("");

  TfLiteTensor* output_tensor = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(output_tensor->bytes, golden_size * sizeof(int16_t));
  int16_t* output = ::tflite::GetTensorData<int16_t>(output_tensor);
  for (uint32_t i = 0; i < golden_size; i++) {
    // TODO(b/205046520): Better understand why TfLite and TFLM can sometimes be
    // off by 1.
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], 1);
  }
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(leaky_relu0_test) {
  tflite::micro::RunModel(
      g_leaky_relu0_model_data, g_leaky_relu0_input0_int16_test_data,
      g_leaky_relu0_input0_int16_test_data_size,
      g_leaky_relu0_golden_int16_test_data,
      g_leaky_relu0_golden_int16_test_data_size, "leaky_relu0 test");
}

TF_LITE_MICRO_TEST(leaky_relu1_test) {
  tflite::micro::RunModel(
      g_leaky_relu1_model_data, g_leaky_relu1_input0_int16_test_data,
      g_leaky_relu1_input0_int16_test_data_size,
      g_leaky_relu1_golden_int16_test_data,
      g_leaky_relu1_golden_int16_test_data_size, "leaky_relu1 test");
}

TF_LITE_MICRO_TEST(leaky_relu2_test) {
  tflite::micro::RunModel(
      g_leaky_relu2_model_data, g_leaky_relu2_input0_int16_test_data,
      g_leaky_relu2_input0_int16_test_data_size,
      g_leaky_relu2_golden_int16_test_data,
      g_leaky_relu2_golden_int16_test_data_size, "leaky_relu2 test");
}

TF_LITE_MICRO_TEST(leaky_relu3_test) {
  tflite::micro::RunModel(
      g_leaky_relu3_model_data, g_leaky_relu3_input0_int16_test_data,
      g_leaky_relu3_input0_int16_test_data_size,
      g_leaky_relu3_golden_int16_test_data,
      g_leaky_relu3_golden_int16_test_data_size, "leaky_relu3 test");
}

TF_LITE_MICRO_TEST(leaky_relu4_test) {
  tflite::micro::RunModel(
      g_leaky_relu4_model_data, g_leaky_relu4_input0_int16_test_data,
      g_leaky_relu4_input0_int16_test_data_size,
      g_leaky_relu4_golden_int16_test_data,
      g_leaky_relu4_golden_int16_test_data_size, "leaky_relu4 test");
}

TF_LITE_MICRO_TEST(leaky_relu5_test) {
  tflite::micro::RunModel(
      g_leaky_relu5_model_data, g_leaky_relu5_input0_int16_test_data,
      g_leaky_relu5_input0_int16_test_data_size,
      g_leaky_relu5_golden_int16_test_data,
      g_leaky_relu5_golden_int16_test_data_size, "leaky_relu5 test");
}

TF_LITE_MICRO_TEST(leaky_relu6_test) {
  tflite::micro::RunModel(
      g_leaky_relu6_model_data, g_leaky_relu6_input0_int16_test_data,
      g_leaky_relu6_input0_int16_test_data_size,
      g_leaky_relu6_golden_int16_test_data,
      g_leaky_relu6_golden_int16_test_data_size, "leaky_relu6 test");
}

TF_LITE_MICRO_TEST(leaky_relu7_test) {
  tflite::micro::RunModel(
      g_leaky_relu7_model_data, g_leaky_relu7_input0_int16_test_data,
      g_leaky_relu7_input0_int16_test_data_size,
      g_leaky_relu7_golden_int16_test_data,
      g_leaky_relu7_golden_int16_test_data_size, "leaky_relu7 test");
}

TF_LITE_MICRO_TEST(leaky_relu8_test) {
  tflite::micro::RunModel(
      g_leaky_relu8_model_data, g_leaky_relu8_input0_int16_test_data,
      g_leaky_relu8_input0_int16_test_data_size,
      g_leaky_relu8_golden_int16_test_data,
      g_leaky_relu8_golden_int16_test_data_size, "leaky_relu8 test");
}

TF_LITE_MICRO_TEST(leaky_relu9_test) {
  tflite::micro::RunModel(
      g_leaky_relu9_model_data, g_leaky_relu9_input0_int16_test_data,
      g_leaky_relu9_input0_int16_test_data_size,
      g_leaky_relu9_golden_int16_test_data,
      g_leaky_relu9_golden_int16_test_data_size, "leaky_relu9 test");
}

TF_LITE_MICRO_TEST(leaky_relu10_test) {
  tflite::micro::RunModel(
      g_leaky_relu10_model_data, g_leaky_relu10_input0_int16_test_data,
      g_leaky_relu10_input0_int16_test_data_size,
      g_leaky_relu10_golden_int16_test_data,
      g_leaky_relu10_golden_int16_test_data_size, "leaky_relu10 test");
}

TF_LITE_MICRO_TEST(leaky_relu11_test) {
  tflite::micro::RunModel(
      g_leaky_relu11_model_data, g_leaky_relu11_input0_int16_test_data,
      g_leaky_relu11_input0_int16_test_data_size,
      g_leaky_relu11_golden_int16_test_data,
      g_leaky_relu11_golden_int16_test_data_size, "leaky_relu11 test");
}

TF_LITE_MICRO_TEST(leaky_relu12_test) {
  tflite::micro::RunModel(
      g_leaky_relu12_model_data, g_leaky_relu12_input0_int16_test_data,
      g_leaky_relu12_input0_int16_test_data_size,
      g_leaky_relu12_golden_int16_test_data,
      g_leaky_relu12_golden_int16_test_data_size, "leaky_relu12 test");
}

TF_LITE_MICRO_TEST(leaky_relu13_test) {
  tflite::micro::RunModel(
      g_leaky_relu13_model_data, g_leaky_relu13_input0_int16_test_data,
      g_leaky_relu13_input0_int16_test_data_size,
      g_leaky_relu13_golden_int16_test_data,
      g_leaky_relu13_golden_int16_test_data_size, "leaky_relu13 test");
}

TF_LITE_MICRO_TEST(leaky_relu14_test) {
  tflite::micro::RunModel(
      g_leaky_relu14_model_data, g_leaky_relu14_input0_int16_test_data,
      g_leaky_relu14_input0_int16_test_data_size,
      g_leaky_relu14_golden_int16_test_data,
      g_leaky_relu14_golden_int16_test_data_size, "leaky_relu14 test");
}

TF_LITE_MICRO_TEST(leaky_relu15_test) {
  tflite::micro::RunModel(
      g_leaky_relu15_model_data, g_leaky_relu15_input0_int16_test_data,
      g_leaky_relu15_input0_int16_test_data_size,
      g_leaky_relu15_golden_int16_test_data,
      g_leaky_relu15_golden_int16_test_data_size, "leaky_relu15 test");
}

TF_LITE_MICRO_TEST(leaky_relu16_test) {
  tflite::micro::RunModel(
      g_leaky_relu16_model_data, g_leaky_relu16_input0_int16_test_data,
      g_leaky_relu16_input0_int16_test_data_size,
      g_leaky_relu16_golden_int16_test_data,
      g_leaky_relu16_golden_int16_test_data_size, "leaky_relu16 test");
}

TF_LITE_MICRO_TEST(leaky_relu17_test) {
  tflite::micro::RunModel(
      g_leaky_relu17_model_data, g_leaky_relu17_input0_int16_test_data,
      g_leaky_relu17_input0_int16_test_data_size,
      g_leaky_relu17_golden_int16_test_data,
      g_leaky_relu17_golden_int16_test_data_size, "leaky_relu17 test");
}

TF_LITE_MICRO_TEST(leaky_relu18_test) {
  tflite::micro::RunModel(
      g_leaky_relu18_model_data, g_leaky_relu18_input0_int16_test_data,
      g_leaky_relu18_input0_int16_test_data_size,
      g_leaky_relu18_golden_int16_test_data,
      g_leaky_relu18_golden_int16_test_data_size, "leaky_relu18 test");
}

TF_LITE_MICRO_TEST(leaky_relu19_test) {
  tflite::micro::RunModel(
      g_leaky_relu19_model_data, g_leaky_relu19_input0_int16_test_data,
      g_leaky_relu19_input0_int16_test_data_size,
      g_leaky_relu19_golden_int16_test_data,
      g_leaky_relu19_golden_int16_test_data_size, "leaky_relu19 test");
}

TF_LITE_MICRO_TEST(leaky_relu20_test) {
  tflite::micro::RunModel(
      g_leaky_relu20_model_data, g_leaky_relu20_input0_int16_test_data,
      g_leaky_relu20_input0_int16_test_data_size,
      g_leaky_relu20_golden_int16_test_data,
      g_leaky_relu20_golden_int16_test_data_size, "leaky_relu20 test");
}

TF_LITE_MICRO_TEST(leaky_relu21_test) {
  tflite::micro::RunModel(
      g_leaky_relu21_model_data, g_leaky_relu21_input0_int16_test_data,
      g_leaky_relu21_input0_int16_test_data_size,
      g_leaky_relu21_golden_int16_test_data,
      g_leaky_relu21_golden_int16_test_data_size, "leaky_relu21 test");
}

TF_LITE_MICRO_TEST(leaky_relu22_test) {
  tflite::micro::RunModel(
      g_leaky_relu22_model_data, g_leaky_relu22_input0_int16_test_data,
      g_leaky_relu22_input0_int16_test_data_size,
      g_leaky_relu22_golden_int16_test_data,
      g_leaky_relu22_golden_int16_test_data_size, "leaky_relu22 test");
}

TF_LITE_MICRO_TESTS_END
