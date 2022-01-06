/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/micro_context.h"

#include <cstdint>

#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

tflite::MicroContext CreateMicroContext(MicroGraph* micro_graph = nullptr) {
  using ::tflite::MicroAllocator;
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();

  const size_t kArenaSize = 1024;
  uint8_t tensor_arena[kArenaSize];
  MicroAllocator* micro_allocator = MicroAllocator::Create(
      tensor_arena, kArenaSize, tflite::GetMicroErrorReporter());

  tflite::MicroContext micro_context(micro_allocator, model, micro_graph);
  return micro_context;
}

// Test structure for external context payload.
struct TestExternalContextPayloadData {
  // Opaque blob
  uint8_t blob_data[128];
};
}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// Ensures that a regular set and get pair works ok.
TF_LITE_MICRO_TEST(TestSetGetExternalContextSuccess) {
  tflite::MicroContext micro_context = tflite::CreateMicroContext();

  tflite::TestExternalContextPayloadData payload;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_context.SetExternalContext(&payload));

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.GetExternalContext());

  // What is returned should be the same as what is set.
  TF_LITE_MICRO_EXPECT((void*)returned_external_context == (void*)(&payload));
}

TF_LITE_MICRO_TEST(TestGetExternalContextWithoutSetShouldReturnNull) {
  tflite::MicroContext micro_context = tflite::CreateMicroContext();

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.GetExternalContext());

  // Return a null if nothing is set before.
  TF_LITE_MICRO_EXPECT((void*)returned_external_context == (nullptr));
}

TF_LITE_MICRO_TEST(TestSetExternalContextCanOnlyBeCalledOnce) {
  tflite::MicroContext micro_context = tflite::CreateMicroContext();

  tflite::TestExternalContextPayloadData payload;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_context.SetExternalContext(&payload));

  // Another set should fail.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_context.SetExternalContext(&payload));
}

TF_LITE_MICRO_TEST(TestGetGraph) {
  tflite::MicroGraph* fake_micro_graph =
      reinterpret_cast<tflite::MicroGraph*>(0xdeadbeef);

  tflite::MicroContext micro_context =
      tflite::CreateMicroContext(fake_micro_graph);

  TF_LITE_MICRO_EXPECT(micro_context.GetGraph() == fake_micro_graph);
}

TF_LITE_MICRO_TESTS_END