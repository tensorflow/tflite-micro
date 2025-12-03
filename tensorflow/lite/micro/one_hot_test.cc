/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/one_hot.h"

#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/one_hot_test_model_data.cc"
#include "tensorflow/lite/schema/schema_generated.h"

using tflite::MicroInterpreter;
using tflite::MicroMutableOpResolver;
using tflite::Model;

extern "C" TfLiteRegistration_V1* Register_ONE_HOT();

extern "C" {
extern const unsigned char g_one_hot_basic_float_model[];
extern const int g_one_hot_basic_float_model_len;
}

namespace tflite {
namespace {}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OneHotBasicFloat) {
  const Model* model = tflite::GetModel(g_one_hot_basic_float_model);
  (const void)model;  // 사용한 것처럼 만들어서 unused 경고 없애기

  // TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());

  // TfLiteTensor* indices = interpreter.input(0);
  // indices->data.i32[0] = 0;
  // indices->data.i32[1] = 1;
  // indices->data.i32[2] = 2;

  // TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  // TfLiteTensor* output = interpreter.output(0);
  // float* out = output->data.f;
  // for 루프로 기대값 비교
}

TF_LITE_MICRO_TESTS_END