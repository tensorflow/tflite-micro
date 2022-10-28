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
#ifndef TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_WRAPPER_H_
#define TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_WRAPPER_H_

#include <Python.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace tflite {

class InterpreterWrapper {
 public:
  InterpreterWrapper(PyObject* model_data,
                     const std::vector<std::string>& registerers_by_name,
                     size_t arena_size, int num_resource_variables);
  ~InterpreterWrapper();

  void PrintAllocations();
  int Invoke();
  int Reset();
  void SetInputTensor(PyObject* data, size_t index);
  PyObject* GetOutputTensor(size_t index) const;
  PyObject* GetInputTensorDetails(size_t index) const;
  PyObject* GetOutputTensorDetails(size_t index) const;

 private:
  tflite::RecordingMicroAllocator* allocator_;
  const PyObject* model_;
  std::unique_ptr<uint8_t[]> memory_arena_;
  tflite::AllOpsResolver all_ops_resolver_;
  tflite::MicroInterpreter* interpreter_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_WRAPPER_H_
