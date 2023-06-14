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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MODEL_RUNNER_MICRO_API_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MODEL_RUNNER_MICRO_API_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.
int micro_model_setup(const void* model_data, int kTensorArenaSize,
                      unsigned char* tensor_arena);

// Runs one iteration of data gathering and inference. This should be called
// repeatedly from the application code. The name needs to be loop() for Arduino
// compatibility.
int micro_model_invoke(unsigned char* input_data, int num_inputs, float* results,
                       int num_outputs, float input_scaling, int zero_bias);

// Returns a pointer to the error reporter
void* get_micro_api_error_reporter();


#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MODEL_RUNNER_MICRO_API_H_
