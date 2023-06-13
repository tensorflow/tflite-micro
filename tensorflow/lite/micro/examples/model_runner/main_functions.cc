/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/examples/model_runner/model.h"
#include "tensorflow/lite/micro/examples/model_runner/output_handler.h"
#include "tensorflow/lite/micro/examples/model_runner/test_data.h"
#include "tensorflow/lite/micro/micro_api.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
// Create an area of memory to use for input, output, and intermediate arrays.\
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 48 * 1024;
// FILL_TENSOR_ARENA_SIZE
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  int ret;

  ret = micro_model_setup(g_model, kTensorArenaSize, tensor_arena);

  if (ret == 1) {
    DebugLog("UNSUPPORTED VERSION.\n");
  } else if (ret == 2) {
    DebugLog("ALLOCATION FAILED.\n");
  }
}

void loop() {
  int result_index = 0;
  float max_value = 0.0f;
  int ret;

  tflite::ErrorReporter* error_reporter =
      (tflite::ErrorReporter*)get_micro_api_error_reporter();

  for (int index = 0; index < TEST_DATA_LENGTH; index++) {
    for (int i = 0; i < MODEL_OUTPUTS; i++) {
      tf_results[i] = 0.0;
    }
    ret = micro_model_invoke(test_data[index], MODEL_INPUTS, results,
                             MODEL_OUTPUTS);

    if (ret == 1) {
      DebugLog("MODEL INVOKE FAILED\n");
      return;
    }

    max_value = results[0];
    result_index = 0;
    for (int i = 1; i < MODEL_OUTPUTS; i++) {
      if (results[i] > max_value) {
        max_value = results[i];
        result_index = i;
      }
    }
    // Produce an output
    HandleOutput(error_reporter, result_index);
  }
  DebugLog("ALL_TESTS_PASSED\n");
}
