/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_COMPATIBILITY_H_
#define TENSORFLOW_LITE_MICRO_COMPATIBILITY_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_log.h"

// C++ will automatically create class-specific delete operators for virtual
// objects, which by default call the global delete function. For embedded
// applications we want to avoid this, and won't be calling new/delete on these
// objects, so we need to override the default implementation with one that does
// nothing to avoid linking in ::delete().
// This macro needs to be included in all subclasses of a virtual base class in
// the private section.
#ifndef TF_LITE_REMOVE_VIRTUAL_DELETE
#define TF_LITE_REMOVE_VIRTUAL_DELETE \
  void operator delete(void* p) {}    \
  void operator delete(void* p, size_t) {}
#endif

#if !defined(TF_LITE_MCU_DEBUG_LOG)
#include <cstdlib>
#ifndef TFLITE_ABORT
#define TFLITE_ABORT abort()
#endif
#else
inline void AbortImpl() {
  MicroPrintf("HALTED");
  while (1) {
  }
}
#ifndef TFLITE_ABORT
#define TFLITE_ABORT AbortImpl();
#endif
#endif

#ifndef TFLITE_ASSERT_FALSE
#if defined(NDEBUG)
#define TFLITE_ASSERT_FALSE (static_cast<void>(0))
#else
#define TFLITE_ASSERT_FALSE TFLITE_ABORT
#endif
#endif

#ifndef TF_LITE_FATAL
#define TF_LITE_FATAL(msg)    \
  do {                        \
    MicroPrintf("%s", (msg)); \
    TFLITE_ABORT;             \
  } while (0)
#endif

#ifndef TF_LITE_ASSERT
#define TF_LITE_ASSERT(x)        \
  do {                           \
    if (!(x)) TF_LITE_FATAL(#x); \
  } while (0)
#endif

#ifndef TFLITE_DCHECK
#define TFLITE_DCHECK(condition) (condition) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_EQ
#define TFLITE_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_NE
#define TFLITE_DCHECK_NE(x, y) ((x) != (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_GE
#define TFLITE_DCHECK_GE(x, y) ((x) >= (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_GT
#define TFLITE_DCHECK_GT(x, y) ((x) > (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_LE
#define TFLITE_DCHECK_LE(x, y) ((x) <= (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_LT
#define TFLITE_DCHECK_LT(x, y) ((x) < (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

// TODO(ahentz): Clean up: We should stick to the DCHECK versions.
#ifndef TFLITE_CHECK
#define TFLITE_CHECK(condition) (condition) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_EQ
#define TFLITE_CHECK_EQ(x, y) ((x) == (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_NE
#define TFLITE_CHECK_NE(x, y) ((x) != (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_GE
#define TFLITE_CHECK_GE(x, y) ((x) >= (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_GT
#define TFLITE_CHECK_GT(x, y) ((x) > (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_LE
#define TFLITE_CHECK_LE(x, y) ((x) <= (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_LT
#define TFLITE_CHECK_LT(x, y) ((x) < (y)) ? (void)0 : TFLITE_ABORT
#endif

// Allow for cross-compiler usage of function signatures - currently used for
// specifying named RUY profiler regions in templated methods.
#ifndef TFLITE_PRETTY_FUNCTION
#if defined(_MSC_VER)
#define TFLITE_PRETTY_FUNCTION __FUNCSIG__
#elif defined(__GNUC__)
#define TFLITE_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#define TFLITE_PRETTY_FUNCTION __func__
#endif
#endif

// TFLITE_DEPRECATED()
//
// Duplicated from absl/base/macros.h to avoid pulling in that library.
// Marks a deprecated class, struct, enum, function, method and variable
// declarations. The macro argument is used as a custom diagnostic message (e.g.
// suggestion of a better alternative).
//
// Example:
//
//   class TFLITE_DEPRECATED("Use Bar instead") Foo {...};
//   TFLITE_DEPRECATED("Use Baz instead") void Bar() {...}
//
// Every usage of a deprecated entity will trigger a warning when compiled with
// clang's `-Wdeprecated-declarations` option. This option is turned off by
// default, but the warnings will be reported by clang-tidy.
#if defined(__clang__) && __cplusplus >= 201103L
#ifndef TFLITE_DEPRECATED
#define TFLITE_DEPRECATED(message) __attribute__((deprecated(message)))
#endif
#endif

#ifndef TFLITE_DEPRECATED
#define TFLITE_DEPRECATED(message)
#endif

#ifndef TFLITE_NOINLINE
#ifdef _WIN32
#define TFLITE_NOINLINE __declspec(noinline)
#else
#ifdef __has_attribute
#if __has_attribute(noinline)
#define TFLITE_NOINLINE __attribute__((noinline))
#else
#define TFLITE_NOINLINE
#endif  // __has_attribute(noinline)
#else
#define TFLITE_NOINLINE
#endif  // __has_attribute
#endif  // _WIN32
#endif  // TFLITE_NOINLINE

#ifndef TFLITE_ATTRIBUTE_WEAK
#if !(defined(__llvm__) && defined(_WIN32)) && !defined(__MINGW32__)
#if defined(__GNUC__) && !defined(__clang__)
#define TFLITE_ATTRIBUTE_WEAK __attribute__((weak))
#elif defined(__has_attribute)
#if __has_attribute(weak)
#define TFLITE_ATTRIBUTE_WEAK __attribute__((weak))
#else
#define TFLITE_ATTRIBUTE_WEAK
#endif  // __has_attribute(weak)
#else
#define TFLITE_ATTRIBUTE_WEAK
#endif
#else
#define TFLITE_ATTRIBUTE_WEAK
#endif
#endif  // TFLITE_ATTRIBUTE_WEAK

#ifndef TFLITE_NO_SANITIZE_INTEGER_OVERFLOW
#if defined(__has_attribute)
#if __has_attribute(no_sanitize)
#if defined(__clang__)
#define TFLITE_NO_SANITIZE_INTEGER_OVERFLOW              \
  __attribute__((no_sanitize("signed-integer-overflow"), \
                 no_sanitize("unsigned-integer-overflow")))
#else
#define TFLITE_NO_SANITIZE_INTEGER_OVERFLOW \
  __attribute__((no_sanitize("signed-integer-overflow")))
#endif
#else
#define TFLITE_NO_SANITIZE_INTEGER_OVERFLOW
#endif
#else
#define TFLITE_NO_SANITIZE_INTEGER_OVERFLOW
#endif
#endif

#endif  // TENSORFLOW_LITE_MICRO_COMPATIBILITY_H_
