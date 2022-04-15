#include "python_error_reporter.h"

namespace tflite {
namespace interpreter_wrapper {

// Report an error message
int PythonErrorReporter::Report(const char* format, va_list args) {
  char buf[1024];
  int formatted = vsnprintf(buf, sizeof(buf), format, args);
  buffer_ << buf;
  return formatted;
}

// // Set's a Python runtime exception with the last error.
// PyObject* PythonErrorReporter::exception() {
//   std::string last_message = message();
//   PyErr_SetString(PyExc_RuntimeError, last_message.c_str());
//   return nullptr;
// }

// // Gets the last error message and clears the buffer.
// std::string PythonErrorReporter::message() {
//   std::string value = buffer_.str();
//   buffer_.clear();
//   return value;
// }
}  // namespace interpreter_wrapper
}  // namespace tflite
