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

#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_input0_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_input0_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv1_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_input0_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv2_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_input0_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv3_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_input0_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_input1_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/transpose_conv/transpose_conv4_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

constexpr size_t kTensorArenaSize = 1024 * 100;
uint8_t tensor_arena[kTensorArenaSize];

namespace tflite {
namespace micro {
namespace {

void RunModel(const uint8_t* model, const int32_t* input0,
              const uint32_t input0_size, const int16_t* input1,
              const uint32_t input1_size, const int16_t* golden,
              const uint32_t golden_size, const char* name) {
  InitializeTarget();
  MicroProfiler profiler;

  MicroInterpreter interpreter(GetModel(model), AllOpsResolver(), tensor_arena,
                               kTensorArenaSize, GetMicroErrorReporter(),
                               nullptr, &profiler);
  interpreter.AllocateTensors();
  TfLiteTensor* input0_tensor = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_EQ(input0_tensor->bytes, input0_size * sizeof(int32_t));
  memcpy(interpreter.input(0)->data.raw, input0, input0_tensor->bytes);
  TfLiteTensor* input1_tensor = interpreter.input(1);
  TF_LITE_MICRO_EXPECT_EQ(input1_tensor->bytes, input1_size * sizeof(int16_t));
  memcpy(interpreter.input(1)->data.raw, input1, input1_tensor->bytes);

  if (kTfLiteOk != interpreter.Invoke()) {
    TF_LITE_MICRO_EXPECT(false);
    return;
  }
  profiler.Log();
  MicroPrintf("");

  TfLiteTensor* output_tensor = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(output_tensor->bytes, golden_size * sizeof(int16_t));
  int16_t* output = output_tensor->data.i16;
  for (uint32_t i = 0; i < golden_size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output[i]);
  }
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(transpose_conv0_test) {
  tflite::micro::RunModel(
      g_transpose_conv0_model_data, g_transpose_conv0_input0_int32_test_data,
      g_transpose_conv0_input0_int32_test_data_size,
      g_transpose_conv0_input1_int16_test_data,
      g_transpose_conv0_input1_int16_test_data_size,
      g_transpose_conv0_golden_int16_test_data,
      g_transpose_conv0_golden_int16_test_data_size, "transpose_conv0 test");
}

TF_LITE_MICRO_TEST(transpose_conv1_test) {
  tflite::micro::RunModel(
      g_transpose_conv1_model_data, g_transpose_conv1_input0_int32_test_data,
      g_transpose_conv1_input0_int32_test_data_size,
      g_transpose_conv1_input1_int16_test_data,
      g_transpose_conv1_input1_int16_test_data_size,
      g_transpose_conv1_golden_int16_test_data,
      g_transpose_conv1_golden_int16_test_data_size, "transpose_conv1 test");
}

TF_LITE_MICRO_TEST(transpose_conv2_test) {
  tflite::micro::RunModel(
      g_transpose_conv2_model_data, g_transpose_conv2_input0_int32_test_data,
      g_transpose_conv2_input0_int32_test_data_size,
      g_transpose_conv2_input1_int16_test_data,
      g_transpose_conv2_input1_int16_test_data_size,
      g_transpose_conv2_golden_int16_test_data,
      g_transpose_conv2_golden_int16_test_data_size, "transpose_conv2 test");
}

TF_LITE_MICRO_TEST(transpose_conv3_test) {
  tflite::micro::RunModel(
      g_transpose_conv3_model_data, g_transpose_conv3_input0_int32_test_data,
      g_transpose_conv3_input0_int32_test_data_size,
      g_transpose_conv3_input1_int16_test_data,
      g_transpose_conv3_input1_int16_test_data_size,
      g_transpose_conv3_golden_int16_test_data,
      g_transpose_conv3_golden_int16_test_data_size, "transpose_conv3 test");
}

TF_LITE_MICRO_TEST(transpose_conv4_test) {
  tflite::micro::RunModel(
      g_transpose_conv4_model_data, g_transpose_conv4_input0_int32_test_data,
      g_transpose_conv4_input0_int32_test_data_size,
      g_transpose_conv4_input1_int16_test_data,
      g_transpose_conv4_input1_int16_test_data_size,
      g_transpose_conv4_golden_int16_test_data,
      g_transpose_conv4_golden_int16_test_data_size, "transpose_conv4 test");
}

TF_LITE_MICRO_TESTS_END
