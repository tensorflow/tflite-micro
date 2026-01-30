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

#include <cstdint>

#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(MicroInterpreterGraphTest, TestResetVariableTensor) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  EXPECT_NE(nullptr, model);

  tflite::testing::TestingOpResolver op_resolver;
  ASSERT_EQ(kTfLiteOk, tflite::testing::GetTestingOpResolver(op_resolver));

  constexpr size_t allocator_buffer_size = 1024 * 16;
  uint8_t allocator_buffer[allocator_buffer_size];

  tflite::MicroInterpreter interpreter(model, op_resolver, allocator_buffer,
                                       allocator_buffer_size, nullptr, nullptr,
                                       true /* preserve_all_tensors */);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // In GetComplexMockModel, tensor index 1 is a variable tensor.
  int variable_tensor_idx = 1;
  TfLiteEvalTensor* variable_tensor =
      interpreter.GetTensor(variable_tensor_idx);
  ASSERT_NE(variable_tensor, nullptr);

  if (variable_tensor->data.data != nullptr) {
    // Fill the variable tensor with non-zero values.
    size_t buffer_size;
    ASSERT_EQ(kTfLiteOk, tflite::TfLiteEvalTensorByteLength(variable_tensor,
                                                            &buffer_size));
    uint8_t* variable_tensor_buffer =
        tflite::micro::GetTensorData<uint8_t>(variable_tensor);
    for (size_t i = 0; i < buffer_size; ++i) {
      variable_tensor_buffer[i] = 0xAA;
    }

    // Reset the variable tensor.
    ASSERT_EQ(kTfLiteOk,
              interpreter.ResetVariableTensor(variable_tensor_idx, 0));

    // Verify that the variable tensor is zeroed out.
    for (size_t i = 0; i < buffer_size; ++i) {
      EXPECT_EQ(0, variable_tensor_buffer[i]);
    }
  }

  // Non-variable tensor should NOT be reset.
  int non_variable_tensor_idx = 0;
  TfLiteEvalTensor* non_variable_tensor =
      interpreter.GetTensor(non_variable_tensor_idx);
  ASSERT_NE(non_variable_tensor, nullptr);
  if (non_variable_tensor->data.data != nullptr) {
    size_t buffer_size;
    ASSERT_EQ(kTfLiteOk, tflite::TfLiteEvalTensorByteLength(non_variable_tensor,
                                                            &buffer_size));
    uint8_t* non_variable_tensor_buffer =
        tflite::micro::GetTensorData<uint8_t>(non_variable_tensor);
    for (size_t i = 0; i < buffer_size; ++i) {
      non_variable_tensor_buffer[i] = 0xBB;
    }
    ASSERT_EQ(kTfLiteError,
              interpreter.ResetVariableTensor(non_variable_tensor_idx, 0));
    for (size_t i = 0; i < buffer_size; ++i) {
      EXPECT_EQ(0xBB, non_variable_tensor_buffer[i]);
    }
  }

  // Test invalid tensor index.
  EXPECT_EQ(kTfLiteError, interpreter.ResetVariableTensor(100, 0));

  // Test invalid subgraph index.
  EXPECT_EQ(kTfLiteError, interpreter.ResetVariableTensor(1, 100));
}

TF_LITE_MICRO_TESTS_MAIN
