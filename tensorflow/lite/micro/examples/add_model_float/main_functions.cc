/*--------------------------------------------------------------------

  main_functions.cc 

  Description: main functions for add_model_float example

  Skyworks Solution
  Copyright (c) 2022, All Rights Reserved

--------------------------------------------------------------------*/

// Hardware has no std out, must use custom handler
//#define NO_COUT

#ifndef NO_COUT
#include <iostream>
#endif

#include "tensorflow/lite/micro/examples/add_model_float/main_functions.h"

//#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/examples/add_model_float/constants.h"
#include "tensorflow/lite/micro/examples/add_model_float/add_model_float_model_data.h"
#include "tensorflow/lite/micro/examples/add_model_float/output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_add_model_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  //static tflite::AllOpsResolver resolver;
  // Pulls in only required OPs, add OP
  static tflite::MicroMutableOpResolver<1> resolver(error_reporter);
  if (resolver.AddAdd() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
  #ifndef NO_COUT
  std::cout << "input dims: " << input->dims->size << std::endl;
  std::cout << "num inputs: " << input->dims->data[0] << std::endl;
  std::cout << "input length: " << input->dims->data[1] << std::endl;
  std::cout << "input type FLOAT32: " << (input->type == kTfLiteFloat32) << std::endl;
  std::cout << "output dims: " << output->dims->size << std::endl;
  std::cout << "num outputs: " << output->dims->data[0] << std::endl;
  std::cout << "output length: " << output->dims->data[1] << std::endl;
  std::cout << "output type FLOAT32: " << (output->type == kTfLiteFloat32) << std::endl;
  #endif

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// creates a random float in range of min and max
float random_float(float min, float max){
  float scale = rand() / (float) RAND_MAX;
  return min + scale * (max - min);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  
  float x = random_float(-kXrange, kXrange);
  input->data.f[0] = x;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }

  float y = output->data.f[0];
  #ifndef NO_COUT
  std::cout << x + x << " ?= " << y << std::endl;
  #else
  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x, y);
  #endif

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
