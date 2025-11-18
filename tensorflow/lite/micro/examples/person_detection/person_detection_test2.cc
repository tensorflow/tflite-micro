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

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/person_detect_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Create an area of memory to use for input, output, and intermediate arrays.
#if defined(XTENSA) && defined(VISION_P6)
constexpr int tensor_arena_size = 352 * 1024;
#else
constexpr int tensor_arena_size = 136 * 1024;
#endif  // defined(XTENSA) && defined(VISION_P6)
uint8_t tensor_arena[tensor_arena_size];

int main(int argc, char** argv) {
  // Parse command-line arguments
  if (argc != 2) {
    printf("ERROR: Incorrect usage.\n");
    printf("Usage: %s <num_invocations>\n", argv[0]);
    return 1;
  }

  int num_invocations = atoi(argv[1]);
  if (num_invocations <= 0) {
    printf("ERROR: Number of invocations must be greater than 0.\n");
    return 1;
  }

  // This is the "startup cost" that the delta measurement will cancel out
  printf("Performing one-time setup...\n");

  // Map the model into a usable data structure
  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    // Don't care
    return 1;
  }

  // Pull in only the operation implementations we need
  tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddDepthwiseConv2D(
      tflite::Register_DEPTHWISE_CONV_2D_INT8());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed.\n");
    return 1;
  }

  // Get information about the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Copy a representative image into the input tensor
  memcpy(input->data.int8, g_person_image_data, input->bytes);

  printf("Setup complete.\n");

  // Run the benchmark loop
  printf("Running %d invocations...\n", num_invocations);

  for (int i = 0; i < num_invocations; ++i) {
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("ERROR: Invoke() failed on iteration %d.\n", i);
      return 1;
    }
  }

  printf("Finished all invocations successfully.\n");

  return 0;
}