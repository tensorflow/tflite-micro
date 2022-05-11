/*--------------------------------------------------------------------

  main_functions.cc 

  Description: main functions for add_model_float example

  Skyworks Solution
  Copyright (c) 2022, All Rights Reserved

--------------------------------------------------------------------*/
// Number of tests to run with randomly generated input
#define NUM_TESTS 100

// Hardware has no std out, must use custom handler
//#define NO_COUT

#ifndef NO_COUT
#include <iostream>
#endif

#include "tensorflow/lite/micro/examples/add_model_float/main_functions.h"
#include <math.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/add_model_float/add_model_float_model_data.h"
#include "tensorflow/lite/micro/examples/add_model_float/output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "constants.h"

// creates a random float in range of min and max
float random_float(float min, float max){
  float scale = rand() / (float) RAND_MAX;
  return min + scale * (max - min);
}

tflite::ErrorReporter* error_reporter = nullptr;

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  // Define the input and the expected output
  //float x = 0.0f;
  float x = random_float(-kXrange, kXrange);
  float y_true = x + x;

  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_add_model_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;

  constexpr int kTensorArenaSize = 2000;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  // Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  // The property "dims" tells us the tensor's shape. It has one element for
  // each dimension. Our input is a 2D tensor containing 1 element, so "dims"
  // should have size 2.
  TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  // The input is an 32 bit float value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  input->data.f[0] = x;

  // Run the model and check that it succeeds
  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Obtain a pointer to the output tensor and make sure it has the
  // properties we expect. It should be the same as the input tensor.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  float y_pred = output->data.f[0];

  // Check if the output is within a small range of the expected output
  float epsilon = 0.00f;
  TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);

  // run tests on randomly generated floats
  for(int test_num = 0; test_num < NUM_TESTS; ++test_num){
    x = random_float(-kXrange, kXrange);
    y_true = x + x;
    input->data.f[0] = x;
    interpreter.Invoke();
    y_pred = output->data.f[0];
    #ifndef NO_COUT
    std::cout << test_num << ": " << y_pred << " ?= " << y_true << std::endl;
    #endif
    TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);
  }
}

TF_LITE_MICRO_TESTS_END
