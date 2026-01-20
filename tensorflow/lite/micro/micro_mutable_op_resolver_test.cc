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

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace {
void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
  // Do nothing.
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {
  // Do nothing.
}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() : has_been_called_(false) {}
  int Report(const char* format, va_list args) override {
    has_been_called_ = true;
    return 0;
  };

  bool HasBeenCalled() { return has_been_called_; }

  void ResetState() { has_been_called_ = false; }

 private:
  bool has_been_called_;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace
}  // namespace tflite

TEST(MicroMutableOpResolverTest, TestOperations) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroMutableOpResolver;

  static TFLMRegistration r = {};
  r.init = tflite::MockInit;
  r.free = tflite::MockFree;
  r.prepare = tflite::MockPrepare;
  r.invoke = tflite::MockInvoke;

  MicroMutableOpResolver<1> micro_op_resolver;
  EXPECT_EQ(kTfLiteOk, micro_op_resolver.AddCustom("mock_custom", &r));

  // Only one AddCustom per operator should return kTfLiteOk.
  EXPECT_EQ(kTfLiteError, micro_op_resolver.AddCustom("mock_custom", &r));

  tflite::MicroOpResolver* resolver = &micro_op_resolver;

  EXPECT_EQ(static_cast<size_t>(1), micro_op_resolver.GetRegistrationLength());

  const TFLMRegistration* registration = resolver->FindOp(BuiltinOperator_RELU);
  EXPECT_EQ(nullptr, registration);

  registration = resolver->FindOp("mock_custom");
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp("nonexistent_custom");
  EXPECT_EQ(nullptr, registration);
}
TF_LITE_MICRO_TESTS_MAIN
