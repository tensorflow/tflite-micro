#include "interpreter_wrapper.h"

#include "python_utils.h"

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

  model = tflite::GetModel(buf);

  static tflite::AllOpsResolver resolver;

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  s_interpreter = &static_interpreter;

  printf("s_i %p\n", s_interpreter);

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
  TfLiteTensor* input = interpreter_->input(0);;
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  input->data.int8[0] = x_quantized;
}

float InterpreterWrapper::GetOutputFloat() {
  TfLiteTensor* output = interpreter_->output(0);;
  int8_t y_quantized = output->data.int8[0];
  return (y_quantized - output->params.zero_point) * output->params.scale;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
