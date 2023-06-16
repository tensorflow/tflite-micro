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
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice0_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice0_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice10_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice10_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice10_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice11_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice11_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice11_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice12_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice12_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice12_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice13_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice13_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice13_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice14_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice14_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice14_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice15_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice15_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice15_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice16_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice16_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice16_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice17_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice17_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice17_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice18_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice18_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice18_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice19_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice19_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice19_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice1_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice1_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice1_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice20_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice20_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice20_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice21_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice21_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice21_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice22_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice22_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice22_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice23_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice23_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice23_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice24_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice24_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice24_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice25_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice25_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice25_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice26_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice26_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice26_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice27_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice27_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice27_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice28_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice28_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice28_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice29_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice29_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice29_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice2_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice2_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice2_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice30_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice30_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice30_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice31_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice31_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice31_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice32_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice32_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice32_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice33_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice33_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice33_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice3_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice3_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice3_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice5_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice5_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice5_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice6_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice6_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice6_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice7_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice7_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice7_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice8_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice8_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice8_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice9_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice9_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice9_model_data.h"
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

TF_LITE_MICRO_TEST(strided_slice0_test) {
  tflite::micro::RunModel(
      g_strided_slice0_model_data, g_strided_slice0_input0_int16_test_data,
      g_strided_slice0_input0_int16_test_data_size,
      g_strided_slice0_golden_int16_test_data,
      g_strided_slice0_golden_int16_test_data_size, "strided_slice0 test");
}

TF_LITE_MICRO_TEST(strided_slice1_test) {
  tflite::micro::RunModel(
      g_strided_slice1_model_data, g_strided_slice1_input0_int16_test_data,
      g_strided_slice1_input0_int16_test_data_size,
      g_strided_slice1_golden_int16_test_data,
      g_strided_slice1_golden_int16_test_data_size, "strided_slice1 test");
}

TF_LITE_MICRO_TEST(strided_slice2_test) {
  tflite::micro::RunModel(
      g_strided_slice2_model_data, g_strided_slice2_input0_int16_test_data,
      g_strided_slice2_input0_int16_test_data_size,
      g_strided_slice2_golden_int16_test_data,
      g_strided_slice2_golden_int16_test_data_size, "strided_slice2 test");
}

TF_LITE_MICRO_TEST(strided_slice3_test) {
  tflite::micro::RunModel(
      g_strided_slice3_model_data, g_strided_slice3_input0_int16_test_data,
      g_strided_slice3_input0_int16_test_data_size,
      g_strided_slice3_golden_int16_test_data,
      g_strided_slice3_golden_int16_test_data_size, "strided_slice3 test");
}

TF_LITE_MICRO_TEST(strided_slice4_test) {
  tflite::micro::RunModel(
      g_strided_slice4_model_data, g_strided_slice4_input0_int16_test_data,
      g_strided_slice4_input0_int16_test_data_size,
      g_strided_slice4_golden_int16_test_data,
      g_strided_slice4_golden_int16_test_data_size, "strided_slice4 test");
}

TF_LITE_MICRO_TEST(strided_slice5_test) {
  tflite::micro::RunModel(
      g_strided_slice5_model_data, g_strided_slice5_input0_int16_test_data,
      g_strided_slice5_input0_int16_test_data_size,
      g_strided_slice5_golden_int16_test_data,
      g_strided_slice5_golden_int16_test_data_size, "strided_slice5 test");
}

TF_LITE_MICRO_TEST(strided_slice6_test) {
  tflite::micro::RunModel(
      g_strided_slice6_model_data, g_strided_slice6_input0_int16_test_data,
      g_strided_slice6_input0_int16_test_data_size,
      g_strided_slice6_golden_int16_test_data,
      g_strided_slice6_golden_int16_test_data_size, "strided_slice6 test");
}

TF_LITE_MICRO_TEST(strided_slice7_test) {
  tflite::micro::RunModel(
      g_strided_slice7_model_data, g_strided_slice7_input0_int16_test_data,
      g_strided_slice7_input0_int16_test_data_size,
      g_strided_slice7_golden_int16_test_data,
      g_strided_slice7_golden_int16_test_data_size, "strided_slice7 test");
}

TF_LITE_MICRO_TEST(strided_slice8_test) {
  tflite::micro::RunModel(
      g_strided_slice8_model_data, g_strided_slice8_input0_int16_test_data,
      g_strided_slice8_input0_int16_test_data_size,
      g_strided_slice8_golden_int16_test_data,
      g_strided_slice8_golden_int16_test_data_size, "strided_slice8 test");
}

TF_LITE_MICRO_TEST(strided_slice9_test) {
  tflite::micro::RunModel(
      g_strided_slice9_model_data, g_strided_slice9_input0_int16_test_data,
      g_strided_slice9_input0_int16_test_data_size,
      g_strided_slice9_golden_int16_test_data,
      g_strided_slice9_golden_int16_test_data_size, "strided_slice9 test");
}

TF_LITE_MICRO_TEST(strided_slice10_test) {
  tflite::micro::RunModel(
      g_strided_slice10_model_data, g_strided_slice10_input0_int16_test_data,
      g_strided_slice10_input0_int16_test_data_size,
      g_strided_slice10_golden_int16_test_data,
      g_strided_slice10_golden_int16_test_data_size, "strided_slice10 test");
}

TF_LITE_MICRO_TEST(strided_slice11_test) {
  tflite::micro::RunModel(
      g_strided_slice11_model_data, g_strided_slice11_input0_int16_test_data,
      g_strided_slice11_input0_int16_test_data_size,
      g_strided_slice11_golden_int16_test_data,
      g_strided_slice11_golden_int16_test_data_size, "strided_slice11 test");
}

TF_LITE_MICRO_TEST(strided_slice12_test) {
  tflite::micro::RunModel(
      g_strided_slice12_model_data, g_strided_slice12_input0_int16_test_data,
      g_strided_slice12_input0_int16_test_data_size,
      g_strided_slice12_golden_int16_test_data,
      g_strided_slice12_golden_int16_test_data_size, "strided_slice12 test");
}

TF_LITE_MICRO_TEST(strided_slice13_test) {
  tflite::micro::RunModel(
      g_strided_slice13_model_data, g_strided_slice13_input0_int16_test_data,
      g_strided_slice13_input0_int16_test_data_size,
      g_strided_slice13_golden_int16_test_data,
      g_strided_slice13_golden_int16_test_data_size, "strided_slice13 test");
}

TF_LITE_MICRO_TEST(strided_slice14_test) {
  tflite::micro::RunModel(
      g_strided_slice14_model_data, g_strided_slice14_input0_int16_test_data,
      g_strided_slice14_input0_int16_test_data_size,
      g_strided_slice14_golden_int16_test_data,
      g_strided_slice14_golden_int16_test_data_size, "strided_slice14 test");
}

TF_LITE_MICRO_TEST(strided_slice15_test) {
  tflite::micro::RunModel(
      g_strided_slice15_model_data, g_strided_slice15_input0_int16_test_data,
      g_strided_slice15_input0_int16_test_data_size,
      g_strided_slice15_golden_int16_test_data,
      g_strided_slice15_golden_int16_test_data_size, "strided_slice15 test");
}

TF_LITE_MICRO_TEST(strided_slice16_test) {
  tflite::micro::RunModel(
      g_strided_slice16_model_data, g_strided_slice16_input0_int16_test_data,
      g_strided_slice16_input0_int16_test_data_size,
      g_strided_slice16_golden_int16_test_data,
      g_strided_slice16_golden_int16_test_data_size, "strided_slice16 test");
}

TF_LITE_MICRO_TEST(strided_slice17_test) {
  tflite::micro::RunModel(
      g_strided_slice17_model_data, g_strided_slice17_input0_int16_test_data,
      g_strided_slice17_input0_int16_test_data_size,
      g_strided_slice17_golden_int16_test_data,
      g_strided_slice17_golden_int16_test_data_size, "strided_slice17 test");
}

TF_LITE_MICRO_TEST(strided_slice18_test) {
  tflite::micro::RunModel(
      g_strided_slice18_model_data, g_strided_slice18_input0_int16_test_data,
      g_strided_slice18_input0_int16_test_data_size,
      g_strided_slice18_golden_int16_test_data,
      g_strided_slice18_golden_int16_test_data_size, "strided_slice18 test");
}

TF_LITE_MICRO_TEST(strided_slice19_test) {
  tflite::micro::RunModel(
      g_strided_slice19_model_data, g_strided_slice19_input0_int16_test_data,
      g_strided_slice19_input0_int16_test_data_size,
      g_strided_slice19_golden_int16_test_data,
      g_strided_slice19_golden_int16_test_data_size, "strided_slice19 test");
}

TF_LITE_MICRO_TEST(strided_slice20_test) {
  tflite::micro::RunModel(
      g_strided_slice20_model_data, g_strided_slice20_input0_int16_test_data,
      g_strided_slice20_input0_int16_test_data_size,
      g_strided_slice20_golden_int16_test_data,
      g_strided_slice20_golden_int16_test_data_size, "strided_slice20 test");
}

TF_LITE_MICRO_TEST(strided_slice21_test) {
  tflite::micro::RunModel(
      g_strided_slice21_model_data, g_strided_slice21_input0_int16_test_data,
      g_strided_slice21_input0_int16_test_data_size,
      g_strided_slice21_golden_int16_test_data,
      g_strided_slice21_golden_int16_test_data_size, "strided_slice21 test");
}

TF_LITE_MICRO_TEST(strided_slice22_test) {
  tflite::micro::RunModel(
      g_strided_slice22_model_data, g_strided_slice22_input0_int16_test_data,
      g_strided_slice22_input0_int16_test_data_size,
      g_strided_slice22_golden_int16_test_data,
      g_strided_slice22_golden_int16_test_data_size, "strided_slice22 test");
}

TF_LITE_MICRO_TEST(strided_slice23_test) {
  tflite::micro::RunModel(
      g_strided_slice23_model_data, g_strided_slice23_input0_int16_test_data,
      g_strided_slice23_input0_int16_test_data_size,
      g_strided_slice23_golden_int16_test_data,
      g_strided_slice23_golden_int16_test_data_size, "strided_slice23 test");
}

TF_LITE_MICRO_TEST(strided_slice24_test) {
  tflite::micro::RunModel(
      g_strided_slice24_model_data, g_strided_slice24_input0_int16_test_data,
      g_strided_slice24_input0_int16_test_data_size,
      g_strided_slice24_golden_int16_test_data,
      g_strided_slice24_golden_int16_test_data_size, "strided_slice24 test");
}

TF_LITE_MICRO_TEST(strided_slice25_test) {
  tflite::micro::RunModel(
      g_strided_slice25_model_data, g_strided_slice25_input0_int16_test_data,
      g_strided_slice25_input0_int16_test_data_size,
      g_strided_slice25_golden_int16_test_data,
      g_strided_slice25_golden_int16_test_data_size, "strided_slice25 test");
}

TF_LITE_MICRO_TEST(strided_slice26_test) {
  tflite::micro::RunModel(
      g_strided_slice26_model_data, g_strided_slice26_input0_int16_test_data,
      g_strided_slice26_input0_int16_test_data_size,
      g_strided_slice26_golden_int16_test_data,
      g_strided_slice26_golden_int16_test_data_size, "strided_slice26 test");
}

TF_LITE_MICRO_TEST(strided_slice27_test) {
  tflite::micro::RunModel(
      g_strided_slice27_model_data, g_strided_slice27_input0_int16_test_data,
      g_strided_slice27_input0_int16_test_data_size,
      g_strided_slice27_golden_int16_test_data,
      g_strided_slice27_golden_int16_test_data_size, "strided_slice27 test");
}

TF_LITE_MICRO_TEST(strided_slice28_test) {
  tflite::micro::RunModel(
      g_strided_slice28_model_data, g_strided_slice28_input0_int16_test_data,
      g_strided_slice28_input0_int16_test_data_size,
      g_strided_slice28_golden_int16_test_data,
      g_strided_slice28_golden_int16_test_data_size, "strided_slice28 test");
}

TF_LITE_MICRO_TEST(strided_slice29_test) {
  tflite::micro::RunModel(
      g_strided_slice29_model_data, g_strided_slice29_input0_int16_test_data,
      g_strided_slice29_input0_int16_test_data_size,
      g_strided_slice29_golden_int16_test_data,
      g_strided_slice29_golden_int16_test_data_size, "strided_slice29 test");
}

TF_LITE_MICRO_TEST(strided_slice30_test) {
  tflite::micro::RunModel(
      g_strided_slice30_model_data, g_strided_slice30_input0_int16_test_data,
      g_strided_slice30_input0_int16_test_data_size,
      g_strided_slice30_golden_int16_test_data,
      g_strided_slice30_golden_int16_test_data_size, "strided_slice30 test");
}

TF_LITE_MICRO_TEST(strided_slice31_test) {
  tflite::micro::RunModel(
      g_strided_slice31_model_data, g_strided_slice31_input0_int16_test_data,
      g_strided_slice31_input0_int16_test_data_size,
      g_strided_slice31_golden_int16_test_data,
      g_strided_slice31_golden_int16_test_data_size, "strided_slice31 test");
}

TF_LITE_MICRO_TEST(strided_slice32_test) {
  tflite::micro::RunModel(
      g_strided_slice32_model_data, g_strided_slice32_input0_int16_test_data,
      g_strided_slice32_input0_int16_test_data_size,
      g_strided_slice32_golden_int16_test_data,
      g_strided_slice32_golden_int16_test_data_size, "strided_slice32 test");
}

TF_LITE_MICRO_TEST(strided_slice33_test) {
  tflite::micro::RunModel(
      g_strided_slice33_model_data, g_strided_slice33_input0_int16_test_data,
      g_strided_slice33_input0_int16_test_data_size,
      g_strided_slice33_golden_int16_test_data,
      g_strided_slice33_golden_int16_test_data_size, "strided_slice33 test");
}

TF_LITE_MICRO_TESTS_END
