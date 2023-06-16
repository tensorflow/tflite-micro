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
#include "tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_golden_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_input0_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/quantize/quantize0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_golden_int32_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_input0_int16_test_data.h"
#include "tensorflow/lite/micro/integration_tests/seanet/quantize/quantize1_model_data.h"
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

template <typename inputT, typename outputT>
void RunModel(const uint8_t* model, const inputT* input0,
              const uint32_t input0_size, const outputT* golden,
              const uint32_t golden_size, const char* name) {
  InitializeTarget();
  MicroProfiler profiler;
  PythonOpsResolver op_resolver;

  MicroInterpreter interpreter(GetModel(model), op_resolver, tensor_arena,
                               kTensorArenaSize, nullptr, &profiler);
  interpreter.AllocateTensors();
  TfLiteTensor* input_tensor0 = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_EQ(input_tensor0->bytes, input0_size * sizeof(inputT));
  memcpy(interpreter.input(0)->data.raw, input0, input_tensor0->bytes);
  if (kTfLiteOk != interpreter.Invoke()) {
    TF_LITE_MICRO_EXPECT(false);
    return;
  }
  if (print_log == true) {
    profiler.Log();
  }
  MicroPrintf("");

  TfLiteTensor* output_tensor = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(output_tensor->bytes, golden_size * sizeof(outputT));
  outputT* output = ::tflite::GetTensorData<outputT>(output_tensor);
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

if (argc > 2) {
  MicroPrintf("wrong number of command line args!\n");
  MicroPrintf("Correct way to run the test is :\n");
  MicroPrintf("if you want to print logs -> ./{PATH TO BINARY} print_logs\n");
  MicroPrintf("if don't want to print logs -> ./{PATH TO BINARY}\n");
} else if ((argc == 2) && (strcmp(argv[1], "print_logs") == 0)) {
  print_log = true;
}

TF_LITE_MICRO_TEST(quantize0_test) {
  tflite::micro::RunModel(
      g_quantize0_model_data, g_quantize0_input0_int32_test_data,
      g_quantize0_input0_int32_test_data_size,
      g_quantize0_golden_int16_test_data,
      g_quantize0_golden_int16_test_data_size, "quantize0 test");
}

TF_LITE_MICRO_TEST(quantize1_test) {
  tflite::micro::RunModel(
      g_quantize1_model_data, g_quantize1_input0_int16_test_data,
      g_quantize1_input0_int16_test_data_size,
      g_quantize1_golden_int32_test_data,
      g_quantize1_golden_int32_test_data_size, "quantize1 test");
}

TF_LITE_MICRO_TESTS_END
