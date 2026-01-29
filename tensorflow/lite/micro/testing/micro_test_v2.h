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

#ifndef TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_V2_H_
#define TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_V2_H_

// A lightweight testing framework designed for use with microcontroller
// applications.
//
// This framework matches the API of GoogleTest but is designed to run on
// systems with minimal library support and no dynamic memory allocation.
//
// Usage:
// ----------------------------------------------------------------------------
// #include "tensorflow/lite/micro/testing/micro_test_v2.h"
//
// TEST(MySuite, MyTest) {
//   EXPECT_EQ(1, 1);
//   ASSERT_TRUE(true);
// }
//
// TF_LITE_MICRO_TESTS_MAIN
// ----------------------------------------------------------------------------

#include <limits>
#include <type_traits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace tflite {
// Initializes the target system for testing. This must be called in main()
// before running any tests.
inline void InitializeTest() { InitializeTarget(); }
}  // namespace tflite

namespace testing {
// Base class for test fixtures. Tests using TEST_F should define a class that
// inherits from this and implements SetUp() and TearDown().
class Test {
 public:
  virtual ~Test() = default;
  virtual void SetUp() {}
  virtual void TearDown() {}
};
}  // namespace testing

namespace micro_test {
namespace internal {

// Information about a registered test case.
struct TestInfo {
  const char* suite_name;
  const char* test_name;
  void (*test_func)();
  TestInfo* next;
  TestInfo*
      next_failure;  // Used to build a list of failed tests during execution.
};

// Global list of registered tests.
inline TestInfo*& GetTestList() {
  static TestInfo* list = nullptr;
  return list;
}

// Helper class to register tests at startup time.
class TestRegistrar {
 public:
  TestRegistrar(TestInfo* info) {
    info->next = GetTestList();
    GetTestList() = info;
  }
};

// Global state to track if the current test has failed.
inline bool& DidTestFail() {
  static bool fail = false;
  return fail;
}

// Overloaded functions to print values of different types.
inline void PrintValue(const char* label, int v) {
  MicroPrintf("%s%d", label, v);
}
inline void PrintValue(const char* label, unsigned int v) {
  MicroPrintf("%s%u", label, v);
}
inline void PrintValue(const char* label, long v) {
  MicroPrintf("%s%ld", label, v);
}
inline void PrintValue(const char* label, unsigned long v) {
  MicroPrintf("%s%lu", label, v);
}
inline void PrintValue(const char* label, float v) {
  MicroPrintf("%s%f", label, static_cast<double>(v));
}
inline void PrintValue(const char* label, double v) {
  MicroPrintf("%s%f", label, v);
}
inline void PrintValue(const char* label, bool v) {
  MicroPrintf("%s%s", label, v ? "true" : "false");
}
inline void PrintValue(const char* label, const char* v) {
  MicroPrintf("%s%s", label, v ? v : "(null)");
}
inline void PrintValue(const char* label, const void* v) {
  MicroPrintf("%s%p", label, v);
}
inline void PrintValue(const char* label, short v) {
  MicroPrintf("%s%d", label, v);
}
inline void PrintValue(const char* label, unsigned short v) {
  MicroPrintf("%s%u", label, v);
}
inline void PrintValue(const char* label, signed char v) {
  MicroPrintf("%s%d", label, v);
}
inline void PrintValue(const char* label, unsigned char v) {
  MicroPrintf("%s%u", label, v);
}
inline void PrintValue(const char* label, long long v) {
  MicroPrintf("%s%lld", label, v);
}
inline void PrintValue(const char* label, unsigned long long v) {
  MicroPrintf("%s%llu", label, static_cast<unsigned long>(v));
}

// Fallback for types that don't match the overloads above.
template <typename T>
inline void PrintValue(const char* label, const T&) {
  MicroPrintf("%s?", label);
}

// Helper to report a failure with the file, line, and comparison details.
template <typename T, typename U>
void ReportFailure(const char* actual_str, const char* expected_str,
                   const T& actual, const U& expected, const char* op,
                   const char* file, int line) {
  MicroPrintf("%s:%d: Failure", file, line);
  MicroPrintf("Value of: %s %s %s", actual_str, op, expected_str);
  PrintValue("  Actual: ", actual);
  PrintValue("Expected: ", expected);
}

// Helper for string equality check.
inline bool AreStringsEqual(const char* s1, const char* s2) {
  if (s1 == s2) return true;
  if (s1 == nullptr || s2 == nullptr) return false;
  while (*s1 && *s2) {
    if (*s1++ != *s2++) return false;
  }
  return *s1 == *s2;
}

}  // namespace internal

// Runs all registered tests and returns kTfLiteOk if all pass, or kTfLiteError
// if any fail.
inline TfLiteStatus RunAllTests() {
  int tests_passed = 0;
  int tests_failed = 0;
  internal::TestInfo* failed_tests = nullptr;
  internal::TestInfo** next_failed_test = &failed_tests;

  // Reverse the list to run tests in the order they were defined.
  internal::TestInfo* prev = nullptr;
  internal::TestInfo* current = internal::GetTestList();
  while (current != nullptr) {
    internal::TestInfo* next = current->next;
    current->next = prev;
    prev = current;
    current = next;
  }
  internal::GetTestList() = prev;

  MicroPrintf("[==========] Running tests.");
  for (internal::TestInfo* test = internal::GetTestList(); test != nullptr;
       test = test->next) {
    MicroPrintf("[ RUN      ] %s.%s", test->suite_name, test->test_name);
    internal::DidTestFail() = false;
    test->test_func();
    if (internal::DidTestFail()) {
      tests_failed++;
      *next_failed_test = test;
      next_failed_test = &test->next_failure;
      test->next_failure = nullptr;
      MicroPrintf("[  FAILED  ] %s.%s", test->suite_name, test->test_name);
    } else {
      tests_passed++;
      MicroPrintf("[       OK ] %s.%s", test->suite_name, test->test_name);
    }
  }

  MicroPrintf("[==========] %d tests ran.", tests_passed + tests_failed);
  MicroPrintf("[  PASSED  ] %d tests.", tests_passed);

  if (tests_failed > 0) {
    MicroPrintf("[  FAILED  ] %d tests, listed below:", tests_failed);
    for (internal::TestInfo* test = failed_tests; test != nullptr;
         test = test->next_failure) {
      MicroPrintf("[  FAILED  ] %s.%s", test->suite_name, test->test_name);
    }
    return kTfLiteError;
  } else {
    // This is for the CI tests expecting this meesage.
    MicroPrintf("~~~ALL TESTS PASSED~~~");
    return kTfLiteOk;
  }
}

}  // namespace micro_test

// -----------------------------------------------------------------------------
// Test Definition Macros
// -----------------------------------------------------------------------------

// Defines a test function.
//
// Arguments:
//   suite: The name of the test suite (e.g. MySuite).
//   name: The name of the test case (e.g. MyTest).
#define TEST(suite, name)                                          \
  void suite##_##name##_Test();                                    \
  static micro_test::internal::TestInfo suite##_##name##_Info = {  \
      #suite, #name, suite##_##name##_Test, nullptr, nullptr};     \
  static micro_test::internal::TestRegistrar suite##_##name##_Reg( \
      &suite##_##name##_Info);                                     \
  void suite##_##name##_Test()

// Defines a test with a fixture.
//
// Arguments:
//   fixture: The name of the fixture class (must inherit from testing::Test).
//   name: The name of the test case.
#define TEST_F(fixture, name)                                                \
  static_assert(std::is_base_of<testing::Test, fixture>::value,              \
                "fixture: The name of the fixture class (must inherit from " \
                "testing::Test).");                                          \
  class fixture##_##name : public fixture {                                  \
   public:                                                                   \
    void TestBody();                                                         \
  };                                                                         \
  void fixture##_##name##_Run() {                                            \
    fixture##_##name test;                                                   \
    test.SetUp();                                                            \
    test.TestBody();                                                         \
    test.TearDown();                                                         \
  }                                                                          \
  static micro_test::internal::TestInfo fixture##_##name##_Info = {          \
      #fixture, #name, fixture##_##name##_Run, nullptr, nullptr};            \
  static micro_test::internal::TestRegistrar fixture##_##name##_Reg(         \
      &fixture##_##name##_Info);                                             \
  void fixture##_##name::TestBody()

// -----------------------------------------------------------------------------
// Internal Helper Macros
// -----------------------------------------------------------------------------

#define MICRO_TEST_BOOL(x, check, msg, on_fail)          \
  do {                                                   \
    if (check) {                                         \
    } else {                                             \
      MicroPrintf("%s:%d: Failure", __FILE__, __LINE__); \
      MicroPrintf("Value of: %s", #x);                   \
      MicroPrintf("Expected: %s", msg);                  \
      micro_test::internal::DidTestFail() = true;        \
      on_fail;                                           \
    }                                                    \
  } while (false)

#define MICRO_TEST_OP(x, y, op_str, compare, on_fail)                       \
  do {                                                                      \
    auto vx = (x);                                                          \
    auto vy = (y);                                                          \
    if (compare) {                                                          \
    } else {                                                                \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, op_str, __FILE__, \
                                          __LINE__);                        \
      micro_test::internal::DidTestFail() = true;                           \
      on_fail;                                                              \
    }                                                                       \
  } while (false)

#define MICRO_TEST_NEAR(x, y, epsilon, on_fail)                 \
  do {                                                          \
    auto vx = (x);                                              \
    auto vy = (y);                                              \
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx)); \
    if (vx != vy && delta > epsilon) {                          \
      MicroPrintf("%s:%d: Failure", __FILE__, __LINE__);        \
      MicroPrintf("Value of: %s near %s", #x, #y);              \
      micro_test::internal::PrintValue("  Actual: ", vx);       \
      micro_test::internal::PrintValue("Expected: ", vy);       \
      micro_test::internal::DidTestFail() = true;               \
      on_fail;                                                  \
    }                                                           \
  } while (false)

// -----------------------------------------------------------------------------
// Assertion Macros (Fatal)
// -----------------------------------------------------------------------------

#define ASSERT_TRUE(x) MICRO_TEST_BOOL(x, x, "was not true", return)

#define ASSERT_FALSE(x) MICRO_TEST_BOOL(x, !(x), "was not false", return)

#define ASSERT_EQ(x, y) MICRO_TEST_OP(x, y, "==", vx == vy, return)

#define ASSERT_NE(x, y) MICRO_TEST_OP(x, y, "!=", vx != vy, return)

#define ASSERT_GT(x, y) MICRO_TEST_OP(x, y, ">", vx > vy, return)

#define ASSERT_LT(x, y) MICRO_TEST_OP(x, y, "<", vx < vy, return)

#define ASSERT_GE(x, y) MICRO_TEST_OP(x, y, ">=", vx >= vy, return)

#define ASSERT_LE(x, y) MICRO_TEST_OP(x, y, "<=", vx <= vy, return)

#define ASSERT_STREQ(x, y)                                                 \
  MICRO_TEST_OP(x, y, "==", micro_test::internal::AreStringsEqual(vx, vy), \
                return)

#define ASSERT_STRNE(x, y)                                                  \
  MICRO_TEST_OP(x, y, "!=", !micro_test::internal::AreStringsEqual(vx, vy), \
                return)

#define ASSERT_FLOAT_EQ(x, y) \
  MICRO_TEST_NEAR(x, y, 4 * std::numeric_limits<float>::epsilon(), return)

#define ASSERT_NEAR(x, y, epsilon) MICRO_TEST_NEAR(x, y, epsilon, return)

// -----------------------------------------------------------------------------
// Expectation Macros (Non-Fatal)
// -----------------------------------------------------------------------------

#define EXPECT_TRUE(x) MICRO_TEST_BOOL(x, x, "was not true", (void)0)

#define EXPECT_FALSE(x) MICRO_TEST_BOOL(x, !(x), "was not false", (void)0)

#define EXPECT_EQ(x, y) MICRO_TEST_OP(x, y, "==", vx == vy, (void)0)

#define EXPECT_NE(x, y) MICRO_TEST_OP(x, y, "!=", vx != vy, (void)0)

#define EXPECT_GT(x, y) MICRO_TEST_OP(x, y, ">", vx > vy, (void)0)

#define EXPECT_LT(x, y) MICRO_TEST_OP(x, y, "<", vx < vy, (void)0)

#define EXPECT_GE(x, y) MICRO_TEST_OP(x, y, ">=", vx >= vy, (void)0)

#define EXPECT_LE(x, y) MICRO_TEST_OP(x, y, "<=", vx <= vy, (void)0)

#define EXPECT_STREQ(x, y)                                                 \
  MICRO_TEST_OP(x, y, "==", micro_test::internal::AreStringsEqual(vx, vy), \
                (void)0)

#define EXPECT_STRNE(x, y)                                                  \
  MICRO_TEST_OP(x, y, "!=", !micro_test::internal::AreStringsEqual(vx, vy), \
                (void)0)

#define EXPECT_NEAR(x, y, epsilon) MICRO_TEST_NEAR(x, y, epsilon, (void)0)

#define EXPECT_FLOAT_EQ(x, y) \
  MICRO_TEST_NEAR(x, y, 4 * std::numeric_limits<float>::epsilon(), (void)0)

// -----------------------------------------------------------------------------
// Other Macros
// -----------------------------------------------------------------------------

#define ADD_FAILURE(msg)                               \
  do {                                                 \
    MicroPrintf("%s:%d: Failure", __FILE__, __LINE__); \
    MicroPrintf("Failed: %s", msg);                    \
    micro_test::internal::DidTestFail() = true;        \
  } while (false)

#define FAIL(msg)                                      \
  do {                                                 \
    MicroPrintf("%s:%d: Failure", __FILE__, __LINE__); \
    MicroPrintf("Failed: %s", msg);                    \
    micro_test::internal::DidTestFail() = true;        \
    return;                                            \
  } while (false)

// Main test runner.
#define RUN_ALL_TESTS() micro_test::RunAllTests()

// Helper macro to create the main function for a test.
#define TF_LITE_MICRO_TESTS_MAIN    \
  int main(int argc, char** argv) { \
    tflite::InitializeTest();       \
    return RUN_ALL_TESTS();         \
  }

#endif  // TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_V2_H_