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

#include "tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {
constexpr int32_t kBufferCnt = 2;
constexpr int32_t kBuffer0Offset = 0;
constexpr int32_t kBuffer1Offset = 10;

// Our c++ convention disallow us to use designated initializers which would
// have simplify the below code to a more readable kOfflineBufferPlan = {
//   .buffer_count = 2,
//   .buffer_plan_entries = {
//       [0] = { .offset = 0 },
//       [1] = { .offset = 10}
//   }
// };
constexpr int32_t kOfflineBufferPlanInBinary[] = {kBufferCnt, kBuffer0Offset,
                                                  kBuffer1Offset};

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestGetOffsetForBuffer) {
  tflite::NonPersistentMemoryPlannerShim planner(
      reinterpret_cast<const tflite::BufferPlan*>(kOfflineBufferPlanInBinary));

  int offset0 = -1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.GetOffsetForBuffer(nullptr, 0, &offset0));
  TF_LITE_MICRO_EXPECT_EQ(kBuffer0Offset, offset0);

  int offset1 = -1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.GetOffsetForBuffer(nullptr, 1, &offset1));
  TF_LITE_MICRO_EXPECT_EQ(kBuffer1Offset, offset1);
}

TF_LITE_MICRO_TEST(TestErrorGetOffsetForBuffer) {
  tflite::NonPersistentMemoryPlannerShim planner(
      reinterpret_cast<const tflite::BufferPlan*>(kOfflineBufferPlanInBinary));

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, planner.GetOffsetForBuffer(nullptr, kBufferCnt, &offset));
}

TF_LITE_MICRO_TEST(TestBasics) {
  tflite::NonPersistentMemoryPlannerShim planner(
      reinterpret_cast<const tflite::BufferPlan*>(kOfflineBufferPlanInBinary));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, planner.AddBuffer(nullptr, 10, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TESTS_END
