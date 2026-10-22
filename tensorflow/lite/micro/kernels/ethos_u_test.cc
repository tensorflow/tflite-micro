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

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/person_detect_vela_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

// The Ethos-U custom operator encapsulates an offline Vela-compiled NPU
// command stream and weight stream for an entire subgraph. Testing the
// operator requires a valid Vela-compiled flatbuffer model. We use the
// frozen person_detect_vela model with deterministic synthetic input tensors
// to avoid circular dependencies on example applications.
constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

TEST(EthosUTest, InvokeVelaModelWithSyntheticData) {
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

  // 1. Evaluate with all-zeros synthetic input.
  std::memset(input->data.int8, 0, input->bytes);
  EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter.output(0);
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output->type, kTfLiteInt8);
  EXPECT_EQ(output->dims->size, 2);
  EXPECT_EQ(output->dims->data[1], 2);

  int8_t zero_out_0 = output->data.int8[0];
  int8_t zero_out_1 = output->data.int8[1];
  MicroPrintf("Synthetic zeros output: [%d, %d]", zero_out_0, zero_out_1);
  EXPECT_GT(zero_out_0, zero_out_1);

  // 2. Evaluate with non-zero synthetic pattern (ramp).
  for (size_t i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = static_cast<int8_t>((i % 255) - 128);
  }
  EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);

  int8_t ramp_out_0 = output->data.int8[0];
  int8_t ramp_out_1 = output->data.int8[1];
  MicroPrintf("Synthetic ramp output: [%d, %d]", ramp_out_0, ramp_out_1);
  EXPECT_GT(ramp_out_1, ramp_out_0);
}

}  // namespace

TF_LITE_MICRO_TESTS_MAIN
