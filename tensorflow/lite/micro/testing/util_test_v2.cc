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

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(Util, ArgumentsExecutedOnlyOnce) {
  float count = 0.;
  // Make sure either argument is executed once after macro expansion.
  EXPECT_NEAR(0, count++, 0.1f);
  EXPECT_NEAR(1, count++, 0.1f);
  EXPECT_NEAR(count++, 2, 0.1f);
  EXPECT_NEAR(count++, 3, 0.1f);
}

TEST(Util, TestExpectEQ) {
  // test EXPECT_EQ for expected behavior
  double a = 2.1;
  EXPECT_EQ(0, 0);
  EXPECT_EQ(true, true);
  EXPECT_EQ(false, false);
  EXPECT_EQ(2.1, a);
  EXPECT_EQ(1.0, true);
  EXPECT_EQ(1.0, 1);
}

TEST(Util, TestExpectNE) {
  // test EXPECT_NE for expected behavior
  float b = 2.1f;
  double a = 2.1;
  EXPECT_NE(0, 1);
  EXPECT_NE(true, false);
  EXPECT_NE(2.10005f, b);
  EXPECT_NE(2.2, a);
}

TF_LITE_MICRO_TESTS_MAIN
