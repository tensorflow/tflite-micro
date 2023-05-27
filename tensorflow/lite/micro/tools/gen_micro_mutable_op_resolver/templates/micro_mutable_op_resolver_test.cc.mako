/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#define VERIFY_OUTPUT ${verify_output}

#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "${path_to_target}/gen_micro_mutable_op_resolver.h"

#include "${target_with_path}_model_data.h"
#if VERIFY_OUTPUT
#include "${target_with_path}_input_${input_dtype}_test_data.h"
#include "${target_with_path}_golden_${output_dtype}_test_data.h"
#endif

constexpr size_t kTensorArenaSize = ${arena_size};
uint8_t tensor_arena[kTensorArenaSize];

namespace tflite {
namespace micro {
namespace {

void RunModel(const uint8_t* model,
              const int8_t* input,
              const uint32_t input_size,
              const int8_t* golden,
              const uint32_t golden_size,
              const char* name) {
  InitializeTarget();
  MicroProfiler profiler;
  tflite::MicroMutableOpResolver<kNumberOperators> op_resolver = get_resolver();

  MicroInterpreter interpreter(GetModel(model), op_resolver, tensor_arena,
                               kTensorArenaSize,
                               nullptr, &profiler);
  interpreter.AllocateTensors();
#if VERIFY_OUTPUT
  TfLiteTensor* input_tensor0 = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_EQ(input_tensor0->bytes,
                          input_size * sizeof(
                              int8_t));
  memcpy(interpreter.input(0)->data.raw,
         input,
         input_tensor0->bytes);
  if (kTfLiteOk != interpreter.Invoke()) {
    TF_LITE_MICRO_EXPECT(false);
    return;
  }
#endif
  profiler.Log();
  MicroPrintf("");

#if VERIFY_OUTPUT
  TfLiteTensor* output_tensor = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(output_tensor->bytes,
                          golden_size * sizeof(int8_t));
  int8_t* output = ::tflite::GetTensorData<int8_t>(output_tensor);
  for (uint32_t i = 0; i < golden_size; i++) {
    // TODO(b/205046520): Better understand why TfLite and TFLM can sometimes be
    // off by 1.
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], 1);
  }
#endif
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(gen_micro_mutable_from_${target}_test) {
#if VERIFY_OUTPUT
tflite::micro::RunModel(
g_${target}_model_data,
g_${target}_input0_${input_dtype}_test_data,
g_${target}_input0_${input_dtype}_test_data_size,
g_${target}_golden_${output_dtype}_test_data,
g_${target}_golden_${output_dtype}_test_data_size,
"${target} test");
#else
tflite::micro::RunModel(
g_${target}_model_data,
nullptr,
0,
nullptr,
0,
"${target} test");
#endif
}

TF_LITE_MICRO_TESTS_END
