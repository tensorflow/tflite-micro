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

#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {
constexpr int32_t kBufferCnt = 2;
constexpr int32_t kBuffer0Offset = 0;
constexpr int32_t kBuffer1Offset = 10;

// Our c++ convention disallow us to use designated initializers which would
// have simplify the below code to a more readable kBufferPlan = {
//   .buffer_count = 2,
//   .buffer_plan_entries = {
//       [0] = { .offset = 0 },
//       [1] = { .offset = 10}
//   }
// };
tflite::BufferPlan* CreateBufferPlan() {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transitent memories in static variables. This is
  // safe because tests are guarateed to run serially.
  alignas(tflite::BufferPlan) static int8_t
      buffer_plan_buffer[tflite::SizeOfBufferPlan(kBufferCnt)];
  tflite::BufferPlan* buffer_plan_ptr =
      new (buffer_plan_buffer) tflite::BufferPlan();
  buffer_plan_ptr->buffer_count = kBufferCnt;
  buffer_plan_ptr->buffer_plan_entries[0].offset = kBuffer0Offset;
  buffer_plan_ptr->buffer_plan_entries[1].offset = kBuffer1Offset;
  return buffer_plan_ptr;
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestGetOffsetForBuffer) {
  tflite::NonPersistentMemoryPlannerShim planner(CreateBufferPlan());

  int offset0 = -1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(0, &offset0));
  TF_LITE_MICRO_EXPECT_EQ(kBuffer0Offset, offset0);

  int offset1 = -1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(1, &offset1));
  TF_LITE_MICRO_EXPECT_EQ(kBuffer1Offset, offset1);
}

TF_LITE_MICRO_TEST(TestErrorGetOffsetForBuffer) {
  tflite::NonPersistentMemoryPlannerShim planner(CreateBufferPlan());

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          planner.GetOffsetForBuffer(kBufferCnt, &offset));
}

TF_LITE_MICRO_TEST(TestAddBufferSuccess) {
  tflite::NonPersistentMemoryPlannerShim planner(CreateBufferPlan());

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, planner.AddBuffer(/*size=*/10,
                                                       /*first_time_used=*/0,
                                                       /*last_time_used=*/1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, planner.AddBuffer(/*size=*/20,
                                                       /*first_time_used=*/0,
                                                       /*last_time_used=*/1));
}

TF_LITE_MICRO_TEST(TestAddBufferFailWhenExceedRange) {
  tflite::NonPersistentMemoryPlannerShim planner(CreateBufferPlan());

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, planner.AddBuffer(/*size=*/10,
                                                       /*first_time_used=*/0,
                                                       /*last_time_used=*/1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, planner.AddBuffer(/*size=*/20,
                                                       /*first_time_used=*/0,
                                                       /*last_time_used=*/1));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError,
      planner.AddBuffer(/*size=*/10,
                        /*first_time_used=*/0, /*last_time_used=*/1));
}

TF_LITE_MICRO_TEST(TestBasics) {
  tflite::NonPersistentMemoryPlannerShim planner(CreateBufferPlan());

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TESTS_END
