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
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv0_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv0_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv10_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv10_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv10_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv11_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv11_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv11_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv12_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv12_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv12_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv13_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv13_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv13_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv14_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv14_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv14_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv15_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv15_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv15_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv16_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv16_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv16_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv17_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv17_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv17_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv18_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv18_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv18_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv19_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv19_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv19_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv1_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv1_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv1_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv20_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv20_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv20_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv21_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv21_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv21_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv2_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv2_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv2_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv3_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv3_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv3_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv4_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv4_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv4_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv5_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv5_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv5_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv6_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv6_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv6_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv7_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv7_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv7_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv8_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv8_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv8_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv9_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv9_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/conv/conv9_model_data.h"
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
              const uint32_t input0_size, const int16_t* golden,
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

TEST(IntegrationTests, conv0_test) {
  tflite::micro::RunModel(g_conv0_model_data, g_conv0_input0_int16_test_data,
                          g_conv0_input0_int16_test_data_size,
                          g_conv0_golden_int16_test_data,
                          g_conv0_golden_int16_test_data_size, "conv0 test");
}

TEST(IntegrationTests, conv1_test) {
  tflite::micro::RunModel(g_conv1_model_data, g_conv1_input0_int16_test_data,
                          g_conv1_input0_int16_test_data_size,
                          g_conv1_golden_int16_test_data,
                          g_conv1_golden_int16_test_data_size, "conv1 test");
}

TEST(IntegrationTests, conv2_test) {
  tflite::micro::RunModel(g_conv2_model_data, g_conv2_input0_int16_test_data,
                          g_conv2_input0_int16_test_data_size,
                          g_conv2_golden_int16_test_data,
                          g_conv2_golden_int16_test_data_size, "conv2 test");
}

TEST(IntegrationTests, conv3_test) {
  tflite::micro::RunModel(g_conv3_model_data, g_conv3_input0_int16_test_data,
                          g_conv3_input0_int16_test_data_size,
                          g_conv3_golden_int16_test_data,
                          g_conv3_golden_int16_test_data_size, "conv3 test");
}

TEST(IntegrationTests, conv4_test) {
  tflite::micro::RunModel(g_conv4_model_data, g_conv4_input0_int16_test_data,
                          g_conv4_input0_int16_test_data_size,
                          g_conv4_golden_int16_test_data,
                          g_conv4_golden_int16_test_data_size, "conv4 test");
}

TEST(IntegrationTests, conv5_test) {
  tflite::micro::RunModel(g_conv5_model_data, g_conv5_input0_int16_test_data,
                          g_conv5_input0_int16_test_data_size,
                          g_conv5_golden_int16_test_data,
                          g_conv5_golden_int16_test_data_size, "conv5 test");
}

TEST(IntegrationTests, conv6_test) {
  tflite::micro::RunModel(g_conv6_model_data, g_conv6_input0_int16_test_data,
                          g_conv6_input0_int16_test_data_size,
                          g_conv6_golden_int16_test_data,
                          g_conv6_golden_int16_test_data_size, "conv6 test");
}

TEST(IntegrationTests, conv7_test) {
  tflite::micro::RunModel(g_conv7_model_data, g_conv7_input0_int16_test_data,
                          g_conv7_input0_int16_test_data_size,
                          g_conv7_golden_int16_test_data,
                          g_conv7_golden_int16_test_data_size, "conv7 test");
}

TEST(IntegrationTests, conv8_test) {
  tflite::micro::RunModel(g_conv8_model_data, g_conv8_input0_int16_test_data,
                          g_conv8_input0_int16_test_data_size,
                          g_conv8_golden_int16_test_data,
                          g_conv8_golden_int16_test_data_size, "conv8 test");
}

TEST(IntegrationTests, conv9_test) {
  tflite::micro::RunModel(g_conv9_model_data, g_conv9_input0_int16_test_data,
                          g_conv9_input0_int16_test_data_size,
                          g_conv9_golden_int16_test_data,
                          g_conv9_golden_int16_test_data_size, "conv9 test");
}

TEST(IntegrationTests, conv10_test) {
  tflite::micro::RunModel(g_conv10_model_data, g_conv10_input0_int16_test_data,
                          g_conv10_input0_int16_test_data_size,
                          g_conv10_golden_int16_test_data,
                          g_conv10_golden_int16_test_data_size, "conv10 test");
}

TEST(IntegrationTests, conv11_test) {
  tflite::micro::RunModel(g_conv11_model_data, g_conv11_input0_int16_test_data,
                          g_conv11_input0_int16_test_data_size,
                          g_conv11_golden_int16_test_data,
                          g_conv11_golden_int16_test_data_size, "conv11 test");
}

TEST(IntegrationTests, conv12_test) {
  tflite::micro::RunModel(g_conv12_model_data, g_conv12_input0_int16_test_data,
                          g_conv12_input0_int16_test_data_size,
                          g_conv12_golden_int16_test_data,
                          g_conv12_golden_int16_test_data_size, "conv12 test");
}

TEST(IntegrationTests, conv13_test) {
  tflite::micro::RunModel(g_conv13_model_data, g_conv13_input0_int16_test_data,
                          g_conv13_input0_int16_test_data_size,
                          g_conv13_golden_int16_test_data,
                          g_conv13_golden_int16_test_data_size, "conv13 test");
}

TEST(IntegrationTests, conv14_test) {
  tflite::micro::RunModel(g_conv14_model_data, g_conv14_input0_int16_test_data,
                          g_conv14_input0_int16_test_data_size,
                          g_conv14_golden_int16_test_data,
                          g_conv14_golden_int16_test_data_size, "conv14 test");
}

TEST(IntegrationTests, conv15_test) {
  tflite::micro::RunModel(g_conv15_model_data, g_conv15_input0_int16_test_data,
                          g_conv15_input0_int16_test_data_size,
                          g_conv15_golden_int16_test_data,
                          g_conv15_golden_int16_test_data_size, "conv15 test");
}

TEST(IntegrationTests, conv16_test) {
  tflite::micro::RunModel(g_conv16_model_data, g_conv16_input0_int16_test_data,
                          g_conv16_input0_int16_test_data_size,
                          g_conv16_golden_int16_test_data,
                          g_conv16_golden_int16_test_data_size, "conv16 test");
}

TEST(IntegrationTests, conv17_test) {
  tflite::micro::RunModel(g_conv17_model_data, g_conv17_input0_int16_test_data,
                          g_conv17_input0_int16_test_data_size,
                          g_conv17_golden_int16_test_data,
                          g_conv17_golden_int16_test_data_size, "conv17 test");
}

TEST(IntegrationTests, conv18_test) {
  tflite::micro::RunModel(g_conv18_model_data, g_conv18_input0_int16_test_data,
                          g_conv18_input0_int16_test_data_size,
                          g_conv18_golden_int16_test_data,
                          g_conv18_golden_int16_test_data_size, "conv18 test");
}

TEST(IntegrationTests, conv19_test) {
  tflite::micro::RunModel(g_conv19_model_data, g_conv19_input0_int16_test_data,
                          g_conv19_input0_int16_test_data_size,
                          g_conv19_golden_int16_test_data,
                          g_conv19_golden_int16_test_data_size, "conv19 test");
}

TEST(IntegrationTests, conv20_test) {
  tflite::micro::RunModel(g_conv20_model_data, g_conv20_input0_int16_test_data,
                          g_conv20_input0_int16_test_data_size,
                          g_conv20_golden_int16_test_data,
                          g_conv20_golden_int16_test_data_size, "conv20 test");
}

TEST(IntegrationTests, conv21_test) {
  tflite::micro::RunModel(g_conv21_model_data, g_conv21_input0_int16_test_data,
                          g_conv21_input0_int16_test_data_size,
                          g_conv21_golden_int16_test_data,
                          g_conv21_golden_int16_test_data_size, "conv21 test");
}

TF_LITE_MICRO_TESTS_MAIN
