/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/models/person_detect_model_data.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/person_detect/gen_micro_mutable_op_resolver.h"

namespace {
#if defined(XTENSA) && defined(VISION_P6)
constexpr int kTensorArenaSize = 352 * 1024;
#else
constexpr int kTensorArenaSize = 136 * 1024;
#endif  // defined(XTENSA) && defined(VISION_P6)
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

TEST(MicroMutableOpResolverTest, PersonDetectModel) {
  tflite::InitializeTarget();
  tflite::MicroMutableOpResolver<kNumberOperators> op_resolver = get_resolver();

  tflite::MicroInterpreter interpreter(
      tflite::GetModel(g_person_detect_model_data), op_resolver, tensor_arena,
      kTensorArenaSize);
  EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  EXPECT_EQ(kTfLiteOk, interpreter.Invoke());
}

TF_LITE_MICRO_TESTS_MAIN
