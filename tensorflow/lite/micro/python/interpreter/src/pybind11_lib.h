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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef TENSORFLOW_LITE_MICRO_PYTHON_INTERPRETER_SRC_PYBIND11_LIB_H_
#define TENSORFLOW_LITE_MICRO_PYTHON_INTERPRETER_SRC_PYBIND11_LIB_H_

namespace py = pybind11;

namespace tflite {

// Convert PyObject* to py::object with no error handling.

inline py::object Pyo(PyObject* ptr) {
  return py::reinterpret_steal<py::object>(ptr);
}

// Raise an exception if the PyErrOccurred flag is set or else return the Python
// object.

inline py::object PyoOrThrow(PyObject* ptr) {
  if (PyErr_Occurred() || ptr == nullptr) {
    throw py::error_already_set();
  }
  return Pyo(ptr);
}

[[noreturn]] inline void ThrowTypeError(const char* error_message) {
  PyErr_SetString(PyExc_TypeError, error_message);
  throw pybind11::error_already_set();
}

[[noreturn]] inline void ThrowValueError(const char* error_message) {
  PyErr_SetString(PyExc_ValueError, error_message);
  throw pybind11::error_already_set();
}

[[noreturn]] inline void ThrowIndexError(const char* error_message) {
  PyErr_SetString(PyExc_IndexError, error_message);
  throw pybind11::error_already_set();
}

[[noreturn]] inline void ThrowRuntimeError(const char* error_message) {
  PyErr_SetString(PyExc_RuntimeError, error_message);
  throw pybind11::error_already_set();
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_PYTHON_INTERPRETER_SRC_PYBIND11_LIB_H_