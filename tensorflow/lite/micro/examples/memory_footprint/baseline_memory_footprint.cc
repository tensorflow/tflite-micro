/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/system_setup.h"

// This file provides a skeleton application without the TFLM Framework
// (interpreter, memory planner etc). This is used to measure the bare minimum
// application code size of a specific target platform without the TFLM
// Framework. Please see README.md for more information.
int main(int argc, char** argv) { tflite::InitializeTarget(); }
