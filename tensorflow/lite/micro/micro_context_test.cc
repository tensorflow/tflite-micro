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

tflite::MicroContext CreateMicroContext() {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transitent memories in static variables. This is
  // safe because tests are guarateed to run serially.
  constexpr size_t kMicroGraphPlacementBufferSize = 1024;
  static uint8_t micro_graph_placement_buffer[kMicroGraphPlacementBufferSize];
  constexpr size_t kArenaSize = 1024;
  static uint8_t tensor_arena[kArenaSize];

  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  MicroAllocator* micro_allocator = MicroAllocator::Create(
      tensor_arena, kArenaSize, tflite::GetMicroErrorReporter());
  MicroGraph* micro_graph = new (micro_graph_placement_buffer)
      MicroGraph(nullptr, nullptr, nullptr, nullptr);

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
                          micro_context.set_external_context(&payload));

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.external_context());

  // What is returned should be the same as what is set.
  TF_LITE_MICRO_EXPECT((void*)returned_external_context == (void*)(&payload));
}

TF_LITE_MICRO_TEST(TestGetExternalContextWithoutSetShouldReturnNull) {
  tflite::MicroContext micro_context = tflite::CreateMicroContext();

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.external_context());

  // Return a null if nothing is set before.
  TF_LITE_MICRO_EXPECT((void*)returned_external_context == (nullptr));
}

TF_LITE_MICRO_TEST(TestSetExternalContextCanOnlyBeCalledOnce) {
  tflite::MicroContext micro_context = tflite::CreateMicroContext();

  tflite::TestExternalContextPayloadData payload;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_context.set_external_context(&payload));

  // Another set should fail.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_context.set_external_context(&payload));
}

TF_LITE_MICRO_TEST(TestSetExternalContextToNullShouldFail) {
  tflite::MicroContext micro_context = tflite::CreateMicroContext();

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_context.set_external_context(nullptr));
}

TF_LITE_MICRO_TESTS_END
