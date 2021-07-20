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

#ifndef XCORE_DISPATCHER_H_
#define XCORE_DISPATCHER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "xcore_persistent_array.h"

#ifdef XCORE

#define ATTRIBUTE_THREAD_FUNCTION                                              \
  __attribute__((fptrgroup("thread_function"), fptrgroup("dispatcher_job")))

#define STRINGIFY_THREAD_FUNCTION(NAME) #NAME

#if RTOS_FREERTOS
// RTOS applications using the RTOS Dispatcher will manage the stack
#define GET_THREAD_FUNCTION_STACKSIZE(DEST, NAME) DEST = 0
#else
// NOTE: In the GET_THREAD_FUNCTION_STACKSIZE macro below, we adjust the stack
// size up by STACKWORDS_ALIGNMENT_ADJUSTMENT.  This is because the
// micro_allocator does not support requests for 8-byte aligned scratch
// buffers.  And, the stack memory pointer for a thread must be 8-byte
// aligned. So, we align it up when necessary. The adjustment makes sure
// we have allocated enough stack memory when the upward alignment is
// necessary.
// TODO: Fix the requirement for this STACKWORDS_ALIGNMENT_ADJUSTMENT someday!
#define STACKWORDS_ALIGNMENT_ADJUSTMENT (2)
#define GET_THREAD_FUNCTION_STACKSIZE(DEST, NAME)                              \
  {                                                                            \
    size_t _stack_words;                                                       \
    asm("ldc %[__dest], " STRINGIFY_THREAD_FUNCTION(NAME) ".nstackwords"       \
        : [ __dest ] "=r"(_stack_words));                                      \
    DEST = (_stack_words + 2 + STACKWORDS_ALIGNMENT_ADJUSTMENT) * 4;           \
  }
#endif

#else // not XCORE

#define ATTRIBUTE_THREAD_FUNCTION
#define GET_THREAD_FUNCTION_STACKSIZE(DEST, NAME) DEST = 0

#endif

namespace tflite {
namespace micro {
namespace xcore {

constexpr size_t kMaxThreads = 5;
constexpr size_t kBytesPerStackword = 4;
constexpr size_t kWordAlignment = 4;
constexpr size_t kDoubleWordAlignment = 8;

using ThreadFunction = void (*)(void *);

class Dispatcher {
public:
  Dispatcher() = default;
  virtual ~Dispatcher() = default;

  /**
   * Initializes the dispatcher
   *
   * @param[in] function      Function to perform, signature must be
   * <tt>void(void*)</tt>
   * @param[in] num_threads   Number of worker threads to use, must be <=
   * maxThreads
   * @param[in] stack_size    Optional single thread worker stack size in bytes
   * (not words)
   * @param[in] stack_memory  Optional stack memory to be used by thread
   * workers. The caller is responsible for allocating enough stack_memory.  The
   * buffer should be >= (stack_size * num_threads).
   * @return                  kTfLiteOk, otherwise kTfLiteError
   */
  template <int maxThreads = kMaxThreads>
  TfLiteStatus Initialize(ThreadFunction function, size_t num_threads,
                          size_t stack_size = 0, char *stack_memory = nullptr) {
    if (num_threads > maxThreads)
      return kTfLiteError;

    function_ = function;
    num_threads_ = num_threads;
    stack_memory_ = stack_memory;
    stack_size_ = stack_size;

    return kTfLiteOk;
  }

  /**
   * Dispatches each item in an array of arguments to a worker thread.
   *
   *   Scheduling is implementation specific, however, an implementation should
   *   guarantee that no more than num_threads_ tasks can be in-flight at any
   *   time.  This method blocks in the caller thread until all arguments have
   *   been invoked and all worker threads have completed processing.
   *
   * @param[in] arguments  Array of worker thread function arguments
   * @param[in] size       Length of arguments array
   *
   * @return               kTfLiteOk, otherwise kTfLiteError
   */
  virtual TfLiteStatus Invoke(void **arguments, size_t size) const = 0;

protected:
  ThreadFunction function_;
  size_t num_threads_;
  char *stack_memory_;
  size_t stack_size_;
};

/**
 * GenericDispatcher class
 *
 * Reference implementation of the Dispatcher abstract base class.
 * Implementations of the GenericDispatcher Invoke() method are provided
 * for XCore and any platform supporting C++ threading.
 */
class GenericDispatcher : public Dispatcher {
public:
  TfLiteStatus Invoke(void **arguments, size_t size) const override;
};

/**
 * Get the global dispatcher.
 *
 * @return    Global Dispatcher object
 */
Dispatcher *GetDispatcher();

/**
 * Set the global dispatcher.
 *
 * @param[in]  dispatcher Pointer to the new global Dispatcher object
 */
void SetDispatcher(Dispatcher *dispatcher);

} // namespace xcore
} // namespace micro
} // namespace tflite

#endif // XCORE_DISPATCHER_H_
