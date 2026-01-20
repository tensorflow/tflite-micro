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
inline void PrintValue(int v) { MicroPrintf("%d", v); }
inline void PrintValue(unsigned int v) { MicroPrintf("%u", v); }
inline void PrintValue(long v) { MicroPrintf("%ld", v); }
inline void PrintValue(unsigned long v) { MicroPrintf("%lu", v); }
inline void PrintValue(float v) { MicroPrintf("%f", static_cast<double>(v)); }
inline void PrintValue(double v) { MicroPrintf("%f", v); }
inline void PrintValue(bool v) { MicroPrintf("%s", v ? "true" : "false"); }
inline void PrintValue(const char* v) { MicroPrintf("%s", v ? v : "(null)"); }
inline void PrintValue(char* v) { MicroPrintf("%s", v ? v : "(null)"); }
inline void PrintValue(const void* v) { MicroPrintf("%p", v); }

// Fallback for types that don't match the overloads above.
template <typename T>
inline void PrintValue(const T&) {
  MicroPrintf("?");
}

// Helper to report a failure with the file, line, and comparison details.
template <typename T, typename U>
void ReportFailure(const char* x_str, const char* y_str, const T& x, const U& y,
                   const char* op, const char* file, int line) {
  MicroPrintf("%s %s %s failed at %s:%d (", x_str, op, y_str, file, line);
  PrintValue(x);
  MicroPrintf(" vs ");
  PrintValue(y);
  MicroPrintf(")\n");
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
inline int RunAllTests() {
  int tests_passed = 0;
  int tests_failed = 0;
  internal::TestInfo* failed_tests = nullptr;
  internal::TestInfo** next_failed_test = &failed_tests;

  for (internal::TestInfo* test = internal::GetTestList(); test != nullptr;
       test = test->next) {
    MicroPrintf("Testing %s.%s", test->suite_name, test->test_name);
    internal::DidTestFail() = false;
    test->test_func();
    if (internal::DidTestFail()) {
      tests_failed++;
      *next_failed_test = test;
      next_failed_test = &test->next_failure;
      test->next_failure = nullptr;
    } else {
      tests_passed++;
    }
  }

  MicroPrintf("%d/%d tests passed", tests_passed,
              (tests_passed + tests_failed));
  if (tests_failed == 0) {
    MicroPrintf("~~~ALL TESTS PASSED~~~\n");
    return kTfLiteOk;
  } else {
    MicroPrintf("~~~SOME TESTS FAILED~~~\n");
    for (internal::TestInfo* test = failed_tests; test != nullptr;
         test = test->next_failure) {
      MicroPrintf("  %s.%s", test->suite_name, test->test_name);
    }
    return kTfLiteError;
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
#define TEST_F(fixture, name)                                        \
  class fixture##_##name : public fixture {                          \
   public:                                                           \
    void TestBody();                                                 \
  };                                                                 \
  void fixture##_##name##_Run() {                                    \
    fixture##_##name test;                                           \
    test.SetUp();                                                    \
    test.TestBody();                                                 \
    test.TearDown();                                                 \
  }                                                                  \
  static micro_test::internal::TestInfo fixture##_##name##_Info = {  \
      #fixture, #name, fixture##_##name##_Run, nullptr, nullptr};    \
  static micro_test::internal::TestRegistrar fixture##_##name##_Reg( \
      &fixture##_##name##_Info);                                     \
  void fixture##_##name::TestBody()

// -----------------------------------------------------------------------------
// Assertion Macros (Fatal)
// -----------------------------------------------------------------------------

#define ASSERT_TRUE(x)                                                     \
  do {                                                                     \
    if (x) {                                                               \
    } else {                                                               \
      MicroPrintf(#x " was not true failed at %s:%d", __FILE__, __LINE__); \
      micro_test::internal::DidTestFail() = true;                          \
      return;                                                              \
    }                                                                      \
  } while (false)

#define ASSERT_FALSE(x)                                                     \
  do {                                                                      \
    if (!(x)) {                                                             \
    } else {                                                                \
      MicroPrintf(#x " was not false failed at %s:%d", __FILE__, __LINE__); \
      micro_test::internal::DidTestFail() = true;                           \
      return;                                                               \
    }                                                                       \
  } while (false)

#define ASSERT_EQ(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx == vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "==", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
      return;                                                             \
    }                                                                     \
  } while (false)

#define ASSERT_NE(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx != vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "!=", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
      return;                                                             \
    }                                                                     \
  } while (false)

#define ASSERT_GT(x, y)                                                  \
  do {                                                                   \
    auto vx = (x);                                                       \
    auto vy = (y);                                                       \
    if (vx > vy) {                                                       \
    } else {                                                             \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, ">", __FILE__, \
                                          __LINE__);                     \
      micro_test::internal::DidTestFail() = true;                        \
      return;                                                            \
    }                                                                    \
  } while (false)

#define ASSERT_LT(x, y)                                                  \
  do {                                                                   \
    auto vx = (x);                                                       \
    auto vy = (y);                                                       \
    if (vx < vy) {                                                       \
    } else {                                                             \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "<", __FILE__, \
                                          __LINE__);                     \
      micro_test::internal::DidTestFail() = true;                        \
      return;                                                            \
    }                                                                    \
  } while (false)

#define ASSERT_GE(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx >= vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, ">=", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
      return;                                                             \
    }                                                                     \
  } while (false)

#define ASSERT_LE(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx <= vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "<=", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
      return;                                                             \
    }                                                                     \
  } while (false)

#define ASSERT_STREQ(x, y)                                                \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (micro_test::internal::AreStringsEqual(vx, vy)) {                  \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "==", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
      return;                                                             \
    }                                                                     \
  } while (false)

#define ASSERT_FLOAT_EQ(x, y)                                                 \
  do {                                                                        \
    auto vx = (x);                                                            \
    auto vy = (y);                                                            \
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));               \
    if (vx != vy && delta > 4 * std::numeric_limits<float>::epsilon()) {      \
      MicroPrintf(#x " (%f) near " #y " (%f) failed at %s:%d",                \
                  static_cast<double>(vx), static_cast<double>(vy), __FILE__, \
                  __LINE__);                                                  \
      micro_test::internal::DidTestFail() = true;                             \
      return;                                                                 \
    }                                                                         \
  } while (false)

// -----------------------------------------------------------------------------
// Expectation Macros (Non-Fatal)
// -----------------------------------------------------------------------------

#define EXPECT_TRUE(x)                                                     \
  do {                                                                     \
    if (x) {                                                               \
    } else {                                                               \
      MicroPrintf(#x " was not true failed at %s:%d", __FILE__, __LINE__); \
      micro_test::internal::DidTestFail() = true;                          \
    }                                                                      \
  } while (false)

#define EXPECT_FALSE(x)                                                     \
  do {                                                                      \
    if (!(x)) {                                                             \
    } else {                                                                \
      MicroPrintf(#x " was not false failed at %s:%d", __FILE__, __LINE__); \
      micro_test::internal::DidTestFail() = true;                           \
    }                                                                       \
  } while (false)

#define EXPECT_EQ(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx == vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "==", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
    }                                                                     \
  } while (false)

#define EXPECT_NE(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx != vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "!=", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
    }                                                                     \
  } while (false)

// Legacy behavior: strict equality for non-floats, epsilon check for floats.
// This is provided to facilitate migration from the old micro_test.h where
// equality check had implicit floating point tolerance.
#define EXPECT_LEGACY_EQ(x, y)                                              \
  do {                                                                      \
    auto vx = (x);                                                          \
    auto vy = (y);                                                          \
    if (vx == vy) {                                                         \
    } else {                                                                \
      bool isFloatingX = (std::is_floating_point<decltype(vx)>::value);     \
      bool isFloatingY = (std::is_floating_point<decltype(vy)>::value);     \
      bool approx_equal = false;                                            \
      if (isFloatingX && isFloatingY) {                                     \
        auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));         \
        if (delta <= std::numeric_limits<decltype(delta)>::epsilon()) {     \
          approx_equal = true;                                              \
        }                                                                   \
      }                                                                     \
      if (!approx_equal) {                                                  \
        micro_test::internal::ReportFailure(#x, #y, vx, vy, "==", __FILE__, \
                                            __LINE__);                      \
        micro_test::internal::DidTestFail() = true;                         \
      }                                                                     \
    }                                                                       \
  } while (false)

#define EXPECT_GT(x, y)                                                  \
  do {                                                                   \
    auto vx = (x);                                                       \
    auto vy = (y);                                                       \
    if (vx > vy) {                                                       \
    } else {                                                             \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, ">", __FILE__, \
                                          __LINE__);                     \
      micro_test::internal::DidTestFail() = true;                        \
    }                                                                    \
  } while (false)

#define EXPECT_LT(x, y)                                                  \
  do {                                                                   \
    auto vx = (x);                                                       \
    auto vy = (y);                                                       \
    if (vx < vy) {                                                       \
    } else {                                                             \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "<", __FILE__, \
                                          __LINE__);                     \
      micro_test::internal::DidTestFail() = true;                        \
    }                                                                    \
  } while (false)

#define EXPECT_GE(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx >= vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, ">=", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
    }                                                                     \
  } while (false)

#define EXPECT_LE(x, y)                                                   \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (vx <= vy) {                                                       \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "<=", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
    }                                                                     \
  } while (false)

#define EXPECT_STREQ(x, y)                                                \
  do {                                                                    \
    auto vx = (x);                                                        \
    auto vy = (y);                                                        \
    if (micro_test::internal::AreStringsEqual(vx, vy)) {                  \
    } else {                                                              \
      micro_test::internal::ReportFailure(#x, #y, vx, vy, "==", __FILE__, \
                                          __LINE__);                      \
      micro_test::internal::DidTestFail() = true;                         \
    }                                                                     \
  } while (false)

#define EXPECT_NEAR(x, y, epsilon)                                            \
  do {                                                                        \
    auto vx = (x);                                                            \
    auto vy = (y);                                                            \
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));               \
    if (vx != vy && delta > epsilon) {                                        \
      MicroPrintf(#x " (%f) near " #y " (%f) failed at %s:%d",                \
                  static_cast<double>(vx), static_cast<double>(vy), __FILE__, \
                  __LINE__);                                                  \
      micro_test::internal::DidTestFail() = true;                             \
    }                                                                         \
  } while (false)

#define EXPECT_FLOAT_EQ(x, y) \
  EXPECT_NEAR(x, y, 4 * std::numeric_limits<float>::epsilon())

#define MICRO_FAIL(msg)                               \
  do {                                                \
    MicroPrintf("FAIL: %s", msg, __FILE__, __LINE__); \
    micro_test::internal::DidTestFail() = true;       \
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
