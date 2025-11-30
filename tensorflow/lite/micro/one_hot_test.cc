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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/one_hot.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
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
  // 테스트케이스별로 추가해달라고 한 패턴
  const Model* model = tflite::GetModel(g_one_hot_basic_float_model);
  (const void)model;  // unused 경고 방지

  // 에러 리포터
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Op 등록 (ONE_HOT만 등록)
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddBuiltin(tflite::BuiltinOperator_ONE_HOT,
                      tflite::ops::micro::Register_ONE_HOT());

  // 인터프리터 생성
  tflite::MicroInterpreter interpreter(model, resolver, g_tensor_arena,
                                       kTensorArenaSize, error_reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());

  // 여기서부터는 g_one_hot_basic_float_model 안에
  // indices / depth / on_value / off_value 가 어떻게 정의되어 있느냐에 따라
  // 입력을 만지거나 그냥 output만 검증하면 됩니다.
  //
  // 예: output[0..8] 이 [1,0,0, 0,1,0, 0,0,1] 이라고 가정하는 경우:
  TfLiteTensor* output = interpreter.output(0);
  float* out_data = output->data.f;

  // 실제 값은 모델에 맞게 바꾸세요.
  TF_LITE_MICRO_EXPECT_EQ(9, output->dims->data[0] * output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_NEAR(1.f, out_data[0], 1e-5f);
}

TF_LITE_MICRO_TESTS_END