#pragma once

#include <Python.h>

// #include "third_party/tensorflow/lite/context.h"
// #include "third_party/tensorflow/lite/string_util.h"

namespace tflite {
namespace python_utils {

int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length);
PyObject* ConvertToPyString(const char* data, size_t length);

}  // namespace python_utils
}  // namespace tflite
