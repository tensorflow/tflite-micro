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

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

TfLiteStatus LoadFloatModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;

  constexpr int kTensorArenaSize = 2056;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensor failed.");
    return kTfLiteError;
  }

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  if (input == nullptr) {
    MicroPrintf("Input tensor in null.");
    return kTfLiteError;
  }

  // Obtain a pointer to the output tensor.
  TfLiteTensor* output = interpreter.output(0);

  // Check if the output is within a small range of the expected output
  float epsilon = 0.05f;

  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    input->data.f[0] = golden_inputs[i];
    interpreter.Invoke();
    float y_pred = output->data.f[0];
    if (abs(sin(golden_inputs[i]) - y_pred) > epsilon) {
      MicroPrintf(
          "Difference between predicted and actual y value "
          "is significant.");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_hello_world_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;

  constexpr int kTensorArenaSize = 2056;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Allocate tensor failed.");
    return kTfLiteError;
  }

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  if (input == nullptr) {
    MicroPrintf("Input tensor in null.");
    return kTfLiteError;
  }

  // Get the input quantization parameters
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Obtain a pointer to the output tensor.
  TfLiteTensor* output = interpreter.output(0);

  // Get the output quantization parameters
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  // Check if the output is within a small range of the expected output
  float epsilon = 0.05f;

  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    input->data.int8[0] = golden_inputs[i] / input_scale + input_zero_point;
    interpreter.Invoke();
    float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
    if (abs(sin(golden_inputs[i]) - y_pred) > epsilon) {
      MicroPrintf(
          "Difference between predicted and actual y value "
          "is significant.");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
