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

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

int LoadFloatModelAndPerformInference() {
  // Define the input and the expected output
  float x = 0.0f;
  float y_true = sin(x);

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

  // Place the quantized input in the model's input tensor
  input->data.f[0] = x;

  // Run the model and check that it succeeds
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Interpreter invocation failed.");
    return kTfLiteError;
  }

  // Obtain a pointer to the output tensor.
  TfLiteTensor* output = interpreter.output(0);

  // Obtain the quantized output from model's output tensor
  float y_pred = output->data.f[0];

  // Check if the output is within a small range of the expected output
  float epsilon = 0.05f;
  if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
  }

  // Run inference on several more values and confirm the expected outputs
  x = 1.f;
  y_true = sin(x);
  input->data.f[0] = x;
  interpreter.Invoke();
  y_pred = output->data.f[0];
  if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
  }

  x = 3.f;
  y_true = sin(x);
  input->data.f[0] = x;
  interpreter.Invoke();
  y_pred = output->data.f[0];
  if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
  }

  x = 5.f;
  y_true = sin(x);
  input->data.f[0] = x;
  interpreter.Invoke();
  y_pred = output->data.f[0];
  if (abs(y_true - y_pred) > epsilon) {
    MicroPrintf(
        "Difference between predicted and actual y value "
        "is significant.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  int status = LoadFloatModelAndPerformInference();
  // To be part of the unit test suite, each test file needs to print out
  // either one of the following strings. These strings are required to
  // be considered as a unit test for the tflm makefiles.
  if (status == kTfLiteOk) {
    MicroPrintf("~~~ALL TESTS PASSED~~~\n");
    return kTfLiteOk;
  } else {
    MicroPrintf("~~~SOME TESTS FAILED~~~\n");
    return kTfLiteError;
  }
}
