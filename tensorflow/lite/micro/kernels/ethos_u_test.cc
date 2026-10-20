/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>

#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/person_detect_vela_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

TEST(EthosUTest, InvokeVelaModel) {
  TFLMRegistration* ethos_u_reg = tflite::Register_ETHOSU();
  if (ethos_u_reg == nullptr) {
    MicroPrintf("Ethos-U is not supported on this platform. Skipping test.");
    return;
  }

  const tflite::Model* model =
      ::tflite::GetModel(g_person_detect_vela_model_data);
  EXPECT_EQ(model->version(), static_cast<uint32_t>(TFLITE_SCHEMA_VERSION));

  tflite::MicroMutableOpResolver<1> resolver;
  EXPECT_EQ(resolver.AddEthosU(), kTfLiteOk);

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);
  EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* input = interpreter.input(0);
  EXPECT_NE(input, nullptr);
  EXPECT_EQ(input->type, kTfLiteInt8);
  EXPECT_EQ(input->bytes, static_cast<size_t>(g_person_image_data_size));

  // 1. Evaluate with person image.
  std::memcpy(input->data.int8, g_person_image_data, input->bytes);
  EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter.output(0);
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output->type, kTfLiteInt8);

  int8_t person_score = output->data.int8[kPersonIndex];
  int8_t no_person_score = output->data.int8[kNotAPersonIndex];
  MicroPrintf("Person image: person=%d, no_person=%d", person_score,
              no_person_score);
  EXPECT_GT(person_score, no_person_score);
  EXPECT_EQ(person_score, 113);
  EXPECT_EQ(no_person_score, -113);

  // 2. Evaluate with no-person image.
  std::memcpy(input->data.int8, g_no_person_image_data, input->bytes);
  EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);

  person_score = output->data.int8[kPersonIndex];
  no_person_score = output->data.int8[kNotAPersonIndex];
  MicroPrintf("No-person image: person=%d, no_person=%d", person_score,
              no_person_score);
  EXPECT_GT(no_person_score, person_score);
  EXPECT_EQ(person_score, -57);
  EXPECT_EQ(no_person_score, 57);
}

}  // namespace

TF_LITE_MICRO_TESTS_MAIN
