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

#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "python/tflite_micro/python_ops_resolver.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

% for target_with_path in targets_with_path:
#include "${target_with_path}_model_data.h"
% for input_idx, input in enumerate(inputs):
#include "${target_with_path}_input${input_idx}_${input_dtypes[input_idx]}_test_data.h"
% endfor
#include "${target_with_path}_golden_${output_dtype}_test_data.h"
% endfor

constexpr size_t kTensorArenaSize = 1024 * 100;
uint8_t tensor_arena[kTensorArenaSize];
bool print_log = false;

namespace tflite {
namespace micro {
namespace {

void RunModel(const uint8_t* model,
% for input_idx, input in enumerate(inputs):
              const ${input_dtypes[input_idx]}_t* input${input_idx},
              const uint32_t input${input_idx}_size,
% endfor
              const ${output_dtype}_t* golden,
              const uint32_t golden_size,
              const char* name) {
  InitializeTarget();
  MicroProfiler profiler;
  PythonOpsResolver op_resolver;

  MicroInterpreter interpreter(GetModel(model), op_resolver, tensor_arena,
                               kTensorArenaSize,
                               nullptr, &profiler);
  interpreter.AllocateTensors();
% for input_idx, input in enumerate(inputs):
  TfLiteTensor* input_tensor${input_idx} = interpreter.input(${input_idx});
  TF_LITE_MICRO_EXPECT_EQ(input_tensor${input_idx}->bytes,
                          input${input_idx}_size * sizeof(
                              ${input_dtypes[input_idx]}_t));
  memcpy(interpreter.input(${input_idx})->data.raw,
         input${input_idx},
         input_tensor${input_idx}->bytes);
% endfor
  if (kTfLiteOk != interpreter.Invoke()) {
    TF_LITE_MICRO_EXPECT(false);
    return;
  }
  if (print_log) {
    profiler.Log();
  }
  MicroPrintf("");

  TfLiteTensor* output_tensor = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(output_tensor->bytes,
                          golden_size * sizeof(${output_dtype}_t));
  ${output_dtype}_t* output = ::tflite::GetTensorData<${output_dtype}_t>(output_tensor);
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

% for target in targets:

TF_LITE_MICRO_TEST(${target}_test) {tflite::micro::RunModel(
g_${target}_model_data,
% for input_idx, input in enumerate(inputs):
g_${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data,
g_${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data_size,
% endfor
g_${target}_golden_${output_dtype}_test_data,
g_${target}_golden_${output_dtype}_test_data_size,
"${target} test");
}

% endfor

TF_LITE_MICRO_TESTS_END
