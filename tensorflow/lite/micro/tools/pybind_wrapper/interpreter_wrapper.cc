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

#include "interpreter_wrapper.h"

#include "python_utils.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace tflite {
namespace interpreter_wrapper {

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* s_interpreter = nullptr;

constexpr int kTensorArenaSize = 20000;
uint8_t tensor_arena[kTensorArenaSize];

// InterpreterWrapper::InterpreterWrapper(const tflite::Model* model,
//                                        tflite::ErrorReporter* error_reporter,
//                                        tflite::AllOpsResolver resolver,
//                                        tflite::MicroInterpreter* interpreter)
//     : model_(model),
//       error_reporter_(std::move(error_reporter)),
//       resolver_(std::move(resolver)),
//       interpreter_(std::move(interpreter)) {
// }

InterpreterWrapper::~InterpreterWrapper() {}

InterpreterWrapper::InterpreterWrapper(PyObject* model_data) {
  char* buf = nullptr;
  Py_ssize_t length;

  printf("-----------------------------\n");

  if (python_utils::ConvertFromPyString(model_data, &buf, &length) == -1) {
    // return nullptr;
    return;
  }

  _import_array();  // TODO: Why not import_array()?

  model = tflite::GetModel(buf);

  static tflite::AllOpsResolver resolver;

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  s_interpreter = &static_interpreter;

  // This doesn't work, why?
  // InterpreterWrapper* wrapper =
  // InterpreterWrapper(model, error_reporter, resolver, s_interpreter);

  model_ = model;
  error_reporter_ = error_reporter;
  resolver_ = resolver;
  interpreter_ = s_interpreter;

  // return wrapper;
}

void InterpreterWrapper::AllocateTensors() {
  TfLiteStatus status = interpreter_->AllocateTensors();
  if (status != kTfLiteOk) {
    printf("AllocateTensors() failed");
  }
}

void InterpreterWrapper::Invoke() {
  TfLiteStatus status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    printf("Invoke() failed");
  }
}

// TODO: Really should be Set/GetTensor
void InterpreterWrapper::SetInputFloat(float x) {
  TfLiteTensor* input = interpreter_->input(0);
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  input->data.int8[0] = x_quantized;
}

float InterpreterWrapper::GetOutputFloat() {
  TfLiteTensor* output = interpreter_->output(0);
  ;
  int8_t y_quantized = output->data.int8[0];
  return (y_quantized - output->params.zero_point) * output->params.scale;
}

void InterpreterWrapper::SetInputTensor(PyObject* data) {
  std::unique_ptr<PyObject, tflite::python_utils::PyDecrefDeleter> array_safe(
      PyArray_FromAny(data, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    return;
  }
  // TODO: error checking
  printf("%p\n", array_safe.get());
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  printf("dims %d %zu", PyArray_NDIM(array), PyArray_NBYTES(array));
  memcpy(interpreter_->input(0)->data.data, PyArray_DATA(array),
         PyArray_NBYTES(array));
}

PyObject* InterpreterWrapper::GetOutputTensor() { return nullptr; }

}  // namespace interpreter_wrapper
}  // namespace tflite
