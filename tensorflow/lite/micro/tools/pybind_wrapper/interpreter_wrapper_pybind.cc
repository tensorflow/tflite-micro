#include <pybind11/pybind11.h>

#include "interpreter_wrapper.h"

namespace py = pybind11;
using tflite::interpreter_wrapper::InterpreterWrapper;

PYBIND11_MODULE(tflm_interpreter, m) {
  m.doc() = "pybind11 TFLM interpreter";

  py::class_<InterpreterWrapper>(m, "InterpreterWrapper")
      .def(py::init([](const py::bytes& data) {
        return std::unique_ptr<InterpreterWrapper>(
            new InterpreterWrapper(data.ptr()));
      }))
      .def("interpreter",
           [](InterpreterWrapper& self) {
             return reinterpret_cast<intptr_t>(self.interpreter());
           })
      .def("AllocateTensors", &InterpreterWrapper::AllocateTensors)
      .def("Invoke", &InterpreterWrapper::Invoke)
      .def(
          "SetInputFloat",
          [](InterpreterWrapper& self, float x) { self.SetInputFloat(x); },
          py::arg("x"))
      .def("GetOutputFloat", &InterpreterWrapper::GetOutputFloat);
}
