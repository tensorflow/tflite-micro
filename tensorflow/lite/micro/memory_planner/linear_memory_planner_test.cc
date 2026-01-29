/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(LinearMemoryPlannerTest, TestBasics) {
  tflite::LinearMemoryPlanner planner;
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(10, 0, 1));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(20, 1, 2));
  EXPECT_EQ(static_cast<size_t>(30), planner.GetMaximumMemorySize());

  int offset = -1;
  EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(0, &offset));
  EXPECT_EQ(0, offset);

  EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(1, &offset));
  EXPECT_EQ(10, offset);
}

TEST(LinearMemoryPlannerTest, TestErrorHandling) {
  tflite::LinearMemoryPlanner planner;
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(10, 0, 1));

  int offset = -1;
  EXPECT_EQ(kTfLiteError, planner.GetOffsetForBuffer(1, &offset));
}

TEST(LinearMemoryPlannerTest, TestPersonDetectionModel) {
  tflite::LinearMemoryPlanner planner;
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(9216, 0, 29));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(3, 28, 29));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(256, 27, 28));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(2304, 26, 27));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(2304, 25, 26));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(2304, 24, 25));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(1152, 23, 24));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 22, 23));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 21, 22));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 20, 21));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 19, 20));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 18, 19));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 17, 18));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 16, 17));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 15, 16));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 14, 15));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 13, 14));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 12, 13));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(2304, 11, 12));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(9216, 10, 11));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(9216, 9, 10));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(9216, 8, 9));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(4608, 7, 8));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(18432, 6, 7));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(18432, 5, 6));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(18432, 4, 5));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(9216, 3, 4));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(36864, 2, 3));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(18432, 1, 2));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(18432, 0, 1));
  EXPECT_EQ(static_cast<size_t>(241027), planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TESTS_MAIN
