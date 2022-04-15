#include "python_utils.h"

#include <memory>

namespace tflite {
namespace python_utils {

int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length) {
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(obj)) {
    // const_cast<> is for CPython 3.7 finally adding const to the API.
    *data = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, length));
    return *data == nullptr ? -1 : 0;
  }
  return PyBytes_AsStringAndSize(obj, data, length);
#else
  return PyString_AsStringAndSize(obj, data, length);
#endif
}

PyObject* ConvertToPyString(const char* data, size_t length) {
#if PY_MAJOR_VERSION >= 3
  return PyBytes_FromStringAndSize(data, length);
#else
  return PyString_FromStringAndSize(data, length);
#endif
}

}  // namespace python_utils
}  // namespace tflite
