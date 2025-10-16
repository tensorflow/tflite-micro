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

#include <stdio.h>
#include <stdlib.h>

#include "tensorflow/lite/micro/examples/person_detection/main_functions.h"

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <number_of_iterations>\n", argv[0]);
    return 1; // Indicate an error
  }

  int loop_count = atoi(argv[1]);
  if (loop_count <= 0) {
    fprintf(stderr, "Error: Please provide a positive number of iterations.\n");
    return 1; // Indicate an error
  }

  setup();

  for (int i = 0; i < loop_count; ++i) {
    loop();
  }
}
