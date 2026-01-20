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

namespace {

// Fixture for testing TEST_F functionality.
class LifecycleTest : public testing::Test {
 public:
  void SetUp() override { setup_count_++; }
  void TearDown() override { teardown_count_++; }

  static int setup_count_;
  static int teardown_count_;
};

int LifecycleTest::setup_count_ = 0;
int LifecycleTest::teardown_count_ = 0;

}  // namespace

// Test that macros evaluate arguments exactly once.
TEST(FrameworkTest, ArgumentsEvaluatedOnce) {
  int counter = 0;
  EXPECT_EQ(++counter, 1);
  ASSERT_EQ(++counter, 2);
  EXPECT_TRUE(++counter == 3);
  ASSERT_TRUE(++counter == 4);
}

// Test boolean expectations.
TEST(FrameworkTest, ExpectBool) {
  EXPECT_TRUE(true);
  EXPECT_FALSE(false);
}

// Test equality expectations.
TEST(FrameworkTest, ExpectEquality) {
  EXPECT_EQ(1, 1);
  EXPECT_EQ(10L, 10L);
  EXPECT_EQ(1.0f, 1.0f);
  EXPECT_EQ(1.0, 1.0);
}

// Test inequality expectations.
TEST(FrameworkTest, ExpectInequality) {
  EXPECT_NE(1, 2);
  EXPECT_NE(1.0f, 2.0f);
}

// Test comparisons (GT, LT, GE, LE).
TEST(FrameworkTest, ExpectComparisons) {
  EXPECT_GT(2, 1);
  EXPECT_GE(2, 1);
  EXPECT_GE(2, 2);

  EXPECT_LT(1, 2);
  EXPECT_LE(1, 2);
  EXPECT_LE(2, 2);
}

// Test string expectations.
TEST(FrameworkTest, ExpectString) {
  const char* str = "hello";
  EXPECT_STREQ(str, "hello");
  EXPECT_STREQ("hello", "hello");
}

// Test floating point approximations.
TEST(FrameworkTest, ExpectFloat) {
  EXPECT_FLOAT_EQ(1.0f, 1.0f);
  EXPECT_FLOAT_EQ(1.0f,
                  1.0f + 1e-7f);  // Within default epsilon (approx 1.19e-7)
  EXPECT_NEAR(1.0f, 1.1f, 0.2f);
}

// Test assertions (fatal failures).
// Note: We can't easily test that they fail and return without a subprocess,
// so we test that they pass when condition matches.
TEST(FrameworkTest, AssertionsPass) {
  ASSERT_TRUE(true);
  ASSERT_FALSE(false);
  ASSERT_EQ(1, 1);
  ASSERT_NE(1, 2);
  ASSERT_GT(2, 1);
  ASSERT_LT(1, 2);
  ASSERT_GE(2, 2);
  ASSERT_LE(2, 2);
  ASSERT_STREQ("foo", "foo");
  ASSERT_FLOAT_EQ(1.0f, 1.0f);
  ASSERT_NEAR(10.0, 10.1, 0.2);
}

// Verify fixture lifecycle execution.
// Since tests run in reverse order of registration (LIFO),
// we verify that SetUp is called and counts increase monotonically.
TEST_F(LifecycleTest, A_TestOne) {
  // Check that SetUp was called for this instance.
  EXPECT_GT(setup_count_, 0);
  // Teardown count lags by one because current test's teardown happens after
  // body.
  EXPECT_EQ(teardown_count_, setup_count_ - 1);
}

TEST_F(LifecycleTest, B_TestTwo) {
  EXPECT_GT(setup_count_, 0);
  EXPECT_EQ(teardown_count_, setup_count_ - 1);
}

TF_LITE_MICRO_TESTS_MAIN
