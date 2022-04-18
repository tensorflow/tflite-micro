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

#pragma once

#include <Python.h>

#include "python_error_reporter.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace interpreter_wrapper {

class InterpreterWrapper {
 public:
  InterpreterWrapper(PyObject* model_data);
  ~InterpreterWrapper();

  tflite::MicroInterpreter* interpreter() { return interpreter_; }

  void AllocateTensors();
  void Invoke();
  void SetInputFloat(float x);
  float GetOutputFloat();
  void SetInputTensor(PyObject* data);
  PyObject* GetOutputTensor();

 private:
  InterpreterWrapper(const tflite::Model* model,
                     tflite::ErrorReporter* error_reporter,
                     tflite::AllOpsResolver resolver,
                     tflite::MicroInterpreter* interpreter);
  const tflite::Model* model_;
  tflite::ErrorReporter* error_reporter_;
  tflite::AllOpsResolver resolver_;
  tflite::MicroInterpreter* interpreter_;
};

}  // namespace interpreter_wrapper
}  // namespace tflite
