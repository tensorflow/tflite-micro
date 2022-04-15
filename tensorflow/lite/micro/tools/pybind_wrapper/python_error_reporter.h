#pragma once

#include <Python.h>

#include <sstream>
#include <string>

#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {
namespace interpreter_wrapper {

class PythonErrorReporter : public tflite::MicroErrorReporter {
 public:
  PythonErrorReporter() {}

  // Report an error message
  int Report(const char* format, va_list args) override;

//   // Sets a Python runtime exception with the last error and
//   // clears the error message buffer.
//   PyObject* exception();

//   // Gets the last error message and clears the buffer.
//   std::string message() override;

 private:
  std::stringstream buffer_;
};

}  // namespace interpreter_wrapper
}
