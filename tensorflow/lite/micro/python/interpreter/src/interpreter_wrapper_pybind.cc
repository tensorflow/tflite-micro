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
#include <pybind11/stl.h>

#include "tensorflow/lite/micro/python/interpreter/src/interpreter_wrapper.h"

namespace py = pybind11;
using tflite::InterpreterWrapper;

PYBIND11_MODULE(interpreter_wrapper_pybind, m) {
  m.doc() = "TFLM interpreter";

  py::class_<InterpreterWrapper>(m, "InterpreterWrapper")
      .def(py::init([](const py::bytes& data,
                       const std::vector<std::string>& registerers_by_name,
                       size_t arena_size) {
        return std::unique_ptr<InterpreterWrapper>(new InterpreterWrapper(
            data.ptr(), registerers_by_name, arena_size));
      }))
      .def("Invoke", &InterpreterWrapper::Invoke)
      .def(
          "SetInputTensor",
          [](InterpreterWrapper& self, py::handle& x, size_t index) {
            self.SetInputTensor(x.ptr(), index);
          },
          py::arg("x"), py::arg("index"))
      .def(
          "GetOutputTensor",
          [](InterpreterWrapper& self, size_t index) {
            return py::reinterpret_steal<py::object>(
                self.GetOutputTensor(index));
          },
          py::arg("index"));
}
