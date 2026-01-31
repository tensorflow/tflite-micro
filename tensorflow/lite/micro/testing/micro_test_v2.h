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

#include <cstdarg>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace tflite {
// Initializes the target system for testing. This must be called in main()
// before running any tests.
inline void InitializeTest() { InitializeTarget(); }
}  // namespace tflite

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

namespace printer {
// Wrapper for DebugLog.
inline void Printf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  DebugLog(format, args);
  va_end(args);
}

// Overloaded functions to print values of different types.
inline void PrintValue(int v) { Printf("%d", v); }
inline void PrintValue(unsigned int v) { Printf("%u", v); }
inline void PrintValue(long v) { Printf("%ld", v); }
inline void PrintValue(unsigned long v) { Printf("%lu", v); }
inline void PrintValue(float v) { Printf("%f", static_cast<double>(v)); }
inline void PrintValue(double v) { Printf("%f", v); }
inline void PrintValue(bool v) { Printf("%s", v ? "true" : "false"); }
inline void PrintValue(const char* v) { Printf("%s", v ? v : "(null)"); }
inline void PrintValue(const void* v) { Printf("%p", v); }
inline void PrintValue(short v) { Printf("%d", v); }
inline void PrintValue(unsigned short v) { Printf("%u", v); }
inline void PrintValue(char v) { Printf("%c", v); }
inline void PrintValue(signed char v) { Printf("%d", v); }
inline void PrintValue(unsigned char v) { Printf("%u", v); }
inline void PrintValue(long long v) { Printf("%lld", v); }
inline void PrintValue(unsigned long long v) { Printf("%llu", v); }
inline void PrintValue(std::nullptr_t) {
  Printf("%p", static_cast<const void*>(nullptr));
}

template <typename T>
inline typename std::enable_if<std::is_enum<T>::value>::type PrintValue(
    const T& v) {
  PrintValue(static_cast<typename std::underlying_type<T>::type>(v));
}

// Fallback for types that don't match the overloads above.
template <typename T>
inline typename std::enable_if<!std::is_enum<T>::value>::type PrintValue(
    const T&) {
  Printf("?");
}

// Helper to report a failure with the file, line, and comparison details.
template <typename T, typename U>
void ReportFailure(const char* a_str, const char* b_str, const T& a, const U& b,
                   const char* op, const char* file, int line) {
  Printf("%s:%d: Failure\n", file, line);
  Printf("  Expression:  %s %s %s\n", a_str, op, b_str);
  Printf("  Evaluated:   ");
  PrintValue(a);
  Printf(" %s ", op);
  PrintValue(b);
  Printf("\n");
}

// Helper to report a near failure.
template <typename T, typename U>
void ReportFailureNear(const char* a_str, const char* b_str,
                       const char* epsilon_str, const T& delta,
                       const U& epsilon, const char* file, int line) {
  Printf("%s:%d: Failure\n", file, line);
  Printf("  Expression: |%s - %s| <= %s\n", a_str, b_str, epsilon_str);
  Printf("  Evaluated:  ");
  PrintValue(delta);
  Printf(" <= ");
  PrintValue(epsilon);
  Printf("\n");
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
}  // namespace printer

// Singleton class to manage test registration and execution.
class TestRunner {
 public:
  static TestRunner& Get() {
    static TestRunner instance;
    return instance;
  }

  void RegisterTest(TestInfo* test) {
    test->next = tests_;
    tests_ = test;
  }

  bool& fail() { return fail_; }
  bool& fatal_fail() { return fatal_fail_; }
  bool& nonfatal_fail() { return nonfatal_fail_; }

  TfLiteStatus Run() {
    int tests_passed = 0;
    int tests_failed = 0;
    TestInfo* failed_tests = nullptr;
    TestInfo** next_failed_test = &failed_tests;

    // Reverse the list to run tests in the order they were defined.
    TestInfo* prev = nullptr;
    TestInfo* current = tests_;
    while (current != nullptr) {
      TestInfo* next = current->next;
      current->next = prev;
      prev = current;
      current = next;
    }
    tests_ = prev;

    printer::Printf("[==========] Running tests.\n");
    for (TestInfo* test = tests_; test != nullptr; test = test->next) {
      printer::Printf("[ RUN      ] %s.%s\n", test->suite_name,
                      test->test_name);
      fail_ = false;
      fatal_fail_ = false;
      nonfatal_fail_ = false;
      test->test_func();
      if (fail_) {
        tests_failed++;
        *next_failed_test = test;
        next_failed_test = &test->next_failure;
        test->next_failure = nullptr;
        printer::Printf("[  FAILED  ] %s.%s\n", test->suite_name,
                        test->test_name);
      } else {
        tests_passed++;
        printer::Printf("[       OK ] %s.%s\n", test->suite_name,
                        test->test_name);
      }
    }

    printer::Printf("[==========] %d tests ran.\n",
                    tests_passed + tests_failed);
    printer::Printf("[  PASSED  ] %d tests.\n", tests_passed);

    if (tests_failed > 0) {
      printer::Printf("[  FAILED  ] %d tests, listed below:\n", tests_failed);
      for (TestInfo* test = failed_tests; test != nullptr;
           test = test->next_failure) {
        printer::Printf("[  FAILED  ] %s.%s\n", test->suite_name,
                        test->test_name);
      }
      return kTfLiteError;
    } else {
      // This is for the CI tests expecting this meesage.
      printer::Printf("~~~ALL TESTS PASSED~~~\n");
      return kTfLiteOk;
    }
  }

 private:
  TestInfo* tests_ = nullptr;
  bool fail_ = false;
  bool fatal_fail_ = false;
  bool nonfatal_fail_ = false;
};

// Helper class to register tests at startup time.
class TestRegistrar {
 public:
  TestRegistrar(TestInfo* info) { TestRunner::Get().RegisterTest(info); }
};

}  // namespace internal

inline bool HasFatalFailure() {
  return internal::TestRunner::Get().fatal_fail();
}
inline bool HasNonfatalFailure() {
  return internal::TestRunner::Get().nonfatal_fail();
}
inline bool HasFailure() { return internal::TestRunner::Get().fail(); }

// Runs all registered tests and returns kTfLiteOk if all pass, or kTfLiteError
// if any fail.
inline TfLiteStatus RunAllTests() { return internal::TestRunner::Get().Run(); }

}  // namespace micro_test

// -----------------------------------------------------------------------------
// Internal Helper Macros
// -----------------------------------------------------------------------------

#define MICRO_TEST_BOOL(a, b, fatal, on_fail)                                 \
  do {                                                                        \
    auto va = (a);                                                            \
    if (va == (b)) {                                                          \
    } else {                                                                  \
      micro_test::internal::printer::ReportFailure(#a, #b, va, b,             \
                                                   "==", __FILE__, __LINE__); \
      micro_test::internal::TestRunner::Get().fail() = true;                  \
      if (fatal) {                                                            \
        micro_test::internal::TestRunner::Get().fatal_fail() = true;          \
      } else {                                                                \
        micro_test::internal::TestRunner::Get().nonfatal_fail() = true;       \
      }                                                                       \
      on_fail;                                                                \
    }                                                                         \
  } while (false)

#define MICRO_TEST_OP(a, b, op_str, compare, fatal, on_fail)               \
  do {                                                                     \
    auto va = (a);                                                         \
    auto vb = (b);                                                         \
    if (compare) {                                                         \
    } else {                                                               \
      micro_test::internal::printer::ReportFailure(#a, #b, va, vb, op_str, \
                                                   __FILE__, __LINE__);    \
      micro_test::internal::TestRunner::Get().fail() = true;               \
      if (fatal) {                                                         \
        micro_test::internal::TestRunner::Get().fatal_fail() = true;       \
      } else {                                                             \
        micro_test::internal::TestRunner::Get().nonfatal_fail() = true;    \
      }                                                                    \
      on_fail;                                                             \
    }                                                                      \
  } while (false)

#define MICRO_TEST_NEAR(a, b, epsilon, fatal, on_fail)                  \
  do {                                                                  \
    auto va = (a);                                                      \
    auto vb = (b);                                                      \
    auto delta = ((va) > (vb)) ? ((va) - (vb)) : ((vb) - (va));         \
    if (va != vb && delta > epsilon) {                                  \
      micro_test::internal::printer::ReportFailureNear(                 \
          #a, #b, #epsilon, delta, epsilon, __FILE__, __LINE__);        \
      micro_test::internal::TestRunner::Get().fail() = true;            \
      if (fatal) {                                                      \
        micro_test::internal::TestRunner::Get().fatal_fail() = true;    \
      } else {                                                          \
        micro_test::internal::TestRunner::Get().nonfatal_fail() = true; \
      }                                                                 \
      on_fail;                                                          \
    }                                                                   \
  } while (false)

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
  static_assert(std::is_base_of<testing::Test, fixture>::value,      \
                "fixture must inherit from testing::Test");          \
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

#define ASSERT_TRUE(x) MICRO_TEST_BOOL(x, true, true, return)

#define ASSERT_FALSE(x) MICRO_TEST_BOOL(x, false, true, return)

#define ASSERT_EQ(a, b) MICRO_TEST_OP(a, b, "==", va == vb, true, return)

#define ASSERT_NE(a, b) MICRO_TEST_OP(a, b, "!=", va != vb, true, return)

#define ASSERT_GT(a, b) MICRO_TEST_OP(a, b, ">", va > vb, true, return)

#define ASSERT_LT(a, b) MICRO_TEST_OP(a, b, "<", va < vb, true, return)

#define ASSERT_GE(a, b) MICRO_TEST_OP(a, b, ">=", va >= vb, true, return)

#define ASSERT_LE(a, b) MICRO_TEST_OP(a, b, "<=", va <= vb, true, return)

#define ASSERT_STREQ(a, b)                                                    \
  MICRO_TEST_OP(a, b,                                                         \
                "==", micro_test::internal::printer::AreStringsEqual(va, vb), \
                true, return)

#define ASSERT_STRNE(a, b)                                                     \
  MICRO_TEST_OP(a, b,                                                          \
                "!=", !micro_test::internal::printer::AreStringsEqual(va, vb), \
                true, return)

#define ASSERT_FLOAT_EQ(a, b) \
  MICRO_TEST_NEAR(a, b, 4 * std::numeric_limits<float>::epsilon(), true, return)

#define ASSERT_NEAR(a, b, epsilon) MICRO_TEST_NEAR(a, b, epsilon, true, return)

#define ASSERT_NO_FATAL_FAILURE(statement)                        \
  do {                                                            \
    bool fatal_failure_before = micro_test::HasFatalFailure();    \
    statement;                                                    \
    if (!fatal_failure_before && micro_test::HasFatalFailure()) { \
      FAIL("Expected no fatal failure, but one occurred.");       \
    }                                                             \
  } while (false)

// -----------------------------------------------------------------------------
// Expectation Macros (Non-Fatal)
// -----------------------------------------------------------------------------

#define EXPECT_TRUE(x) MICRO_TEST_BOOL(x, true, false, (void)0)

#define EXPECT_FALSE(x) MICRO_TEST_BOOL(x, false, false, (void)0)

#define EXPECT_EQ(a, b) MICRO_TEST_OP(a, b, "==", va == vb, false, (void)0)

#define EXPECT_NE(a, b) MICRO_TEST_OP(a, b, "!=", va != vb, false, (void)0)

#define EXPECT_GT(a, b) MICRO_TEST_OP(a, b, ">", va > vb, false, (void)0)

#define EXPECT_LT(a, b) MICRO_TEST_OP(a, b, "<", va < vb, false, (void)0)

#define EXPECT_GE(a, b) MICRO_TEST_OP(a, b, ">=", va >= vb, false, (void)0)

#define EXPECT_LE(a, b) MICRO_TEST_OP(a, b, "<=", va <= vb, false, (void)0)

#define EXPECT_STREQ(a, b)                                                    \
  MICRO_TEST_OP(a, b,                                                         \
                "==", micro_test::internal::printer::AreStringsEqual(va, vb), \
                false, (void)0)

#define EXPECT_STRNE(a, b)                                                     \
  MICRO_TEST_OP(a, b,                                                          \
                "!=", !micro_test::internal::printer::AreStringsEqual(va, vb), \
                false, (void)0)

#define EXPECT_NEAR(a, b, epsilon) \
  MICRO_TEST_NEAR(a, b, epsilon, false, (void)0)

#define EXPECT_FLOAT_EQ(a, b)                                             \
  MICRO_TEST_NEAR(a, b, 4 * std::numeric_limits<float>::epsilon(), false, \
                  (void)0)

#define EXPECT_NO_FATAL_FAILURE(statement)                         \
  do {                                                             \
    bool fatal_failure_before = micro_test::HasFatalFailure();     \
    statement;                                                     \
    if (!fatal_failure_before && micro_test::HasFatalFailure()) {  \
      ADD_FAILURE("Expected no fatal failure, but one occurred."); \
    }                                                              \
  } while (false)

// -----------------------------------------------------------------------------
// Other Macros
// -----------------------------------------------------------------------------

#define ADD_FAILURE(msg)                                                \
  do {                                                                  \
    micro_test::internal::printer::Printf("%s:%d: Failure\n", __FILE__, \
                                          __LINE__);                    \
    micro_test::internal::printer::Printf("Failed: %s\n", msg);         \
    micro_test::internal::TestRunner::Get().fail() = true;              \
    micro_test::internal::TestRunner::Get().nonfatal_fail() = true;     \
  } while (false)

#define FAIL(msg)                                                       \
  do {                                                                  \
    micro_test::internal::printer::Printf("%s:%d: Failure\n", __FILE__, \
                                          __LINE__);                    \
    micro_test::internal::printer::Printf("Failed: %s\n", msg);         \
    micro_test::internal::TestRunner::Get().fail() = true;              \
    micro_test::internal::TestRunner::Get().fatal_fail() = true;        \
    return;                                                             \
  } while (false)

// Main test runner.
#define RUN_ALL_TESTS() micro_test::RunAllTests()

// Helper macro to create the main function for a test.
#define TF_LITE_MICRO_TESTS_MAIN    \
  int main(int argc, char** argv) { \
    tflite::InitializeTest();       \
    return RUN_ALL_TESTS();         \
  }

// -----------------------------------------------------------------------------
// Global accessors for test failures (googletest compatibility)
// -----------------------------------------------------------------------------

namespace testing {
// Base class for test fixtures. Tests using TEST_F should define a class that
// inherits from this and implements SetUp() and TearDown().
class Test {
 public:
  virtual ~Test() = default;
  virtual void SetUp() {}
  virtual void TearDown() {}

  static bool HasFatalFailure() { return micro_test::HasFatalFailure(); }
  static bool HasNonfatalFailure() { return micro_test::HasNonfatalFailure(); }
  static bool HasFailure() { return micro_test::HasFailure(); }
};
}  // namespace testing

inline bool HasFatalFailure() { return micro_test::HasFatalFailure(); }
inline bool HasNonfatalFailure() { return micro_test::HasNonfatalFailure(); }
inline bool HasFailure() { return micro_test::HasFailure(); }

#endif  // TENSORFLOW_LITE_MICRO_TESTING_MICRO_TEST_V2_H_