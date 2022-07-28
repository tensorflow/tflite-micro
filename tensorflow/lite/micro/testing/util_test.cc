/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ArgumentsExecutedOnlyOnce) {
  float count = 0.;
  // Make sure either argument is executed once after macro expansion.
  TF_LITE_MICRO_EXPECT_NEAR(0, count++, 0.1f);
  TF_LITE_MICRO_EXPECT_NEAR(1, count++, 0.1f);
  TF_LITE_MICRO_EXPECT_NEAR(count++, 2, 0.1f);
  TF_LITE_MICRO_EXPECT_NEAR(count++, 3, 0.1f);
}

TF_LITE_MICRO_TEST(TestExpectEQ) {
  // test TF_LITE_EXPECT_EQ for expected behavior
  double a = 2.1;
  TF_LITE_MICRO_EXPECT_EQ(0, 0);
  TF_LITE_MICRO_EXPECT_EQ(true, true);
  TF_LITE_MICRO_EXPECT_EQ(false, false);
  TF_LITE_MICRO_EXPECT_EQ(2.1, a);
  TF_LITE_MICRO_EXPECT_EQ(1.0, true);
  TF_LITE_MICRO_EXPECT_EQ(1.0, 1);
}

TF_LITE_MICRO_TEST(TestExpectNE) {
  // test TF_LITE_EXPECT_NE for expected behavior
  float b = 2.1f;
  double a = 2.1;
  TF_LITE_MICRO_EXPECT_NE(0, 1);
  TF_LITE_MICRO_EXPECT_NE(true, false);
  TF_LITE_MICRO_EXPECT_NE(2.10005f, b);
  TF_LITE_MICRO_EXPECT_NE(2.2, a);
}

TF_LITE_MICRO_TESTS_END
