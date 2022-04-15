#pragma once

#include <Python.h>

#include "python_error_reporter.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace interpreter_wrapper {

class InterpreterWrapper {
 public:
  InterpreterWrapper(PyObject* model_data);
  ~InterpreterWrapper();

  tflite::MicroInterpreter* interpreter() { return interpreter_; }

  void AllocateTensors();
  void Invoke();
  void SetInputFloat(float x);
  float GetOutputFloat();

 private:
  InterpreterWrapper(const tflite::Model* model,
                     tflite::ErrorReporter* error_reporter,
                     tflite::AllOpsResolver resolver,
                     tflite::MicroInterpreter* interpreter);
  const tflite::Model* model_;
  tflite::ErrorReporter* error_reporter_;
  tflite::AllOpsResolver resolver_;
  tflite::MicroInterpreter* interpreter_;
};

}  // namespace interpreter_wrapper
}  // namespace tflite
