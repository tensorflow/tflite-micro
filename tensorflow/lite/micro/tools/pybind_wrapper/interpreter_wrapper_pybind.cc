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

#include <pybind11/pybind11.h>

#include "interpreter_wrapper.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace py = pybind11;
using tflite::interpreter_wrapper::InterpreterWrapper;

void* enable_numpy_support() {
  Py_Initialize();
  import_array();
  return nullptr;
}

PYBIND11_MODULE(tflm_interpreter, m) {
  m.doc() = "pybind11 TFLM interpreter";

  // enable_numpy_support();

  py::class_<InterpreterWrapper>(m, "InterpreterWrapper")
      .def(py::init([](const py::bytes& data) {
        return std::unique_ptr<InterpreterWrapper>(
            new InterpreterWrapper(data.ptr()));  // TODO where is this deleted?
      }))
      .def("interpreter",
           [](InterpreterWrapper& self) {
             return reinterpret_cast<intptr_t>(self.interpreter());
           })
      .def("AllocateTensors", &InterpreterWrapper::AllocateTensors)
      .def("Invoke", &InterpreterWrapper::Invoke)
      // .def(
      //     "SetInputFloat",
      //     [](InterpreterWrapper& self, float x) { self.SetInputFloat(x); },
      //     py::arg("x"))
      // .def("GetOutputFloat", &InterpreterWrapper::GetOutputFloat)
      .def(
          "SetInputTensor",
          [](InterpreterWrapper& self, py::handle& x, int index) {
            self.SetInputTensor(x.ptr(), index);
          },
          py::arg("x"), py::arg("index"))
      .def("GetOutputTensor", [](InterpreterWrapper& self) {
        return py::reinterpret_steal<py::object>(
            self.GetOutputTensor());  // Pyo
      });
}
