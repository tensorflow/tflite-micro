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

int g_eval_counter = 0;
int IncrementCounter() { return ++g_eval_counter; }

}  // namespace

TEST(FrameworkTest, ArgumentsEvaluatedOnce) {
  g_eval_counter = 0;
  EXPECT_EQ(IncrementCounter(), 1);
  ASSERT_EQ(IncrementCounter(), 2);
  EXPECT_TRUE(IncrementCounter() == 3);
  ASSERT_TRUE(IncrementCounter() == 4);
}

TEST(FrameworkTest, BooleanExpectations) {
  EXPECT_TRUE(true);
  EXPECT_TRUE(!false);
  EXPECT_FALSE(false);
  EXPECT_FALSE(!true);
}

TEST(FrameworkTest, IntegerEquality) {
  EXPECT_EQ(1, 1);
  const int kInt = 123;
  EXPECT_EQ(kInt, 123);

  EXPECT_EQ(10L, 10L);
  EXPECT_EQ(100LL, 100LL);

  EXPECT_EQ(1u, 1u);
  EXPECT_EQ(10ul, 10ul);
  EXPECT_EQ(100ull, 100ull);

  EXPECT_EQ(static_cast<short>(5), static_cast<short>(5));
  EXPECT_EQ(static_cast<unsigned short>(5), static_cast<unsigned short>(5));
  EXPECT_EQ(static_cast<signed char>(5), static_cast<signed char>(5));
  EXPECT_EQ(static_cast<unsigned char>(5), static_cast<unsigned char>(5));
}

TEST(FrameworkTest, ComparisonExpectations) {
  EXPECT_NE(1, 2);
  EXPECT_GT(2, 1);
  EXPECT_GE(2, 1);
  EXPECT_GE(2, 2);
  EXPECT_LT(1, 2);
  EXPECT_LE(1, 2);
  EXPECT_LE(2, 2);
}

TEST(FrameworkTest, StringExpectations) {
  const char* str = "hello";
  EXPECT_STREQ(str, "hello");
  EXPECT_STREQ("hello", "hello");

  char buffer[] = "hello";
  EXPECT_STREQ(buffer, "hello");

  EXPECT_STRNE(str, "world");
  EXPECT_STRNE("hello", "world");
}

TEST(FrameworkTest, FloatingPointExpectations) {
  EXPECT_FLOAT_EQ(1.0f, 1.0f);
  EXPECT_FLOAT_EQ(1.0f, 1.0000001f);

  EXPECT_NEAR(1.0f, 1.1f, 0.2f);
  EXPECT_NEAR(1.0, 1.1, 0.2);
}

TEST(FrameworkTest, PointerExpectations) {
  int x = 10;
  int* p1 = &x;
  int* p2 = &x;
  EXPECT_EQ(p1, p2);

  int y = 10;
  EXPECT_NE(&x, &y);
}

TEST(FrameworkTest, FatalAssertionsPass) {
  ASSERT_TRUE(true);
  ASSERT_FALSE(false);
  ASSERT_EQ(1, 1);
  ASSERT_NE(1, 2);
  ASSERT_GT(2, 1);
  ASSERT_LT(1, 2);
  ASSERT_GE(2, 2);
  ASSERT_LE(2, 2);
  ASSERT_STREQ("foo", "foo");
  ASSERT_STRNE("foo", "bar");
  ASSERT_FLOAT_EQ(1.0f, 1.0f);
  ASSERT_NEAR(10.0, 10.1, 0.2);
}

TEST(FrameworkTest, FailureAccessors) {
  EXPECT_FALSE(HasFailure());
  EXPECT_FALSE(HasFatalFailure());
  EXPECT_FALSE(HasNonfatalFailure());

  EXPECT_EQ(1, 1);
  EXPECT_FALSE(HasFailure());
}

TEST(FrameworkTest, NoFatalFailure) {
  ASSERT_NO_FATAL_FAILURE(EXPECT_TRUE(true));
  EXPECT_NO_FATAL_FAILURE(EXPECT_TRUE(true));
  ASSERT_NO_FATAL_FAILURE(ASSERT_TRUE(true));
  EXPECT_NO_FATAL_FAILURE(ASSERT_TRUE(true));
}

namespace {

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

TEST_F(LifecycleTest, TestInstance1) {
  EXPECT_GT(setup_count_, 0);
  EXPECT_EQ(teardown_count_, setup_count_ - 1);
  EXPECT_FALSE(HasFailure());
  EXPECT_FALSE(testing::Test::HasFailure());
}

TEST_F(LifecycleTest, TestInstance2) {
  EXPECT_GT(setup_count_, 0);
  EXPECT_EQ(teardown_count_, setup_count_ - 1);
}

TF_LITE_MICRO_TESTS_MAIN
