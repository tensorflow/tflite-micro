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

#include "tensorflow/lite/micro/testing/test_all_op_resolver.h"

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace {

void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

}  // namespace

TEST(TestAllOpResolverTest, DefaultResolverFindsBuiltinOps) {
  TestAllOpResolver<> resolver;

  // Verify representative builtin operators are registered and found.
  EXPECT_NE(resolver.FindOp(BuiltinOperator_ABS), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_ADD), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_CONV_2D), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_DEPTHWISE_CONV_2D), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_FULLY_CONNECTED), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_RESHAPE), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_SOFTMAX), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_ZEROS_LIKE), nullptr);

  // All builtin ops are registered (103 without Ethos-U, 104 with Ethos-U).
  EXPECT_GE(resolver.GetRegistrationLength(), 103u);
  EXPECT_LE(resolver.GetRegistrationLength(), kBuiltinOpsCount);
}

TEST(TestAllOpResolverTest, ResolverWithExtraSlotsSupportsCustomOps) {
  // Sized with 2 extra custom op slots.
  TestAllOpResolver<2> resolver;

  // Builtin ops still present.
  EXPECT_NE(resolver.FindOp(BuiltinOperator_ADD), nullptr);

  static TFLMRegistration mock_reg = {};
  mock_reg.init = MockInit;
  mock_reg.free = MockFree;
  mock_reg.prepare = MockPrepare;
  mock_reg.invoke = MockInvoke;

  unsigned int initial_len = resolver.GetRegistrationLength();
  EXPECT_EQ(resolver.AddCustom("mock_custom_1", &mock_reg), kTfLiteOk);
  EXPECT_NE(resolver.FindOp("mock_custom_1"), nullptr);
  EXPECT_EQ(resolver.GetRegistrationLength(), initial_len + 1);

  EXPECT_EQ(resolver.AddCustom("mock_custom_2", &mock_reg), kTfLiteOk);
  EXPECT_NE(resolver.FindOp("mock_custom_2"), nullptr);
  EXPECT_EQ(resolver.GetRegistrationLength(), initial_len + 2);

  // Exceeding capacity fails.
  // Note: if AddEthosU() didn't register on this platform, there may be 1 extra
  // slot from the Ethos-U reservation. We consume any remaining slots until
  // full.
  static const char* kExtraOpNames[] = {"mock_custom_3", "mock_custom_4"};
  int index = 0;
  while (resolver.GetRegistrationLength() < kBuiltinOpsCount + 2) {
    EXPECT_EQ(resolver.AddCustom(kExtraOpNames[index++], &mock_reg), kTfLiteOk);
  }
  EXPECT_EQ(resolver.AddCustom("mock_custom_overflow", &mock_reg),
            kTfLiteError);
}

TEST(TestAllOpResolverTest, StandaloneRegisterBuiltinOps) {
  MicroMutableOpResolver<kBuiltinOpsCount> resolver;
  EXPECT_EQ(RegisterBuiltinOps(resolver), kTfLiteOk);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_ADD), nullptr);
  EXPECT_NE(resolver.FindOp(BuiltinOperator_CONV_2D), nullptr);
}

}  // namespace tflite

TF_LITE_MICRO_TESTS_MAIN
