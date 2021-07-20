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

#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"

#ifdef XCORE

extern "C" {
#include <xcore/thread.h>
}

#else

#include <utility>
#include <vector>
#include <thread>

#endif

#include "tensorflow/lite/micro/memory_helpers.h"

namespace tflite {
namespace micro {
namespace xcore {

static Dispatcher* kDispatcher = nullptr;

void SetDispatcher(Dispatcher* dispatcher) { kDispatcher = dispatcher; }

Dispatcher* GetDispatcher() {
  assert(kDispatcher);
  return kDispatcher;
}

#ifdef XCORE

TfLiteStatus GenericDispatcher::Invoke(void** arguments, size_t size) const {
  threadgroup_t group = thread_group_alloc();

  size_t stack_words = stack_size_ / kBytesPerStackword;

  // Align up the stack pointer (if necessary)
  uint8_t* aligned_stack_start = AlignPointerUp(
      reinterpret_cast<uint8_t*>(stack_memory_), kDoubleWordAlignment);

  size_t num_threads_added = 0;
  uint8_t* aligned_stack = aligned_stack_start;

  for (int i = 0; i < size; i++) {
    thread_group_add(group, function_, arguments[i],
                     stack_base(aligned_stack, stack_words));

    aligned_stack =
        AlignPointerUp(&aligned_stack[stack_size_], kDoubleWordAlignment);
    num_threads_added += 1;

    if (num_threads_added == num_threads_) {
      // Dispatch tasks
      // NOTE: This implementation processes the tasks in groups of num_threads.
      //       This performs optimally only when the tasks have similar
      //       complexity. Future implementations may no longer require tasks
      //       of similar complexity in order to dispatch the work optimally.
      thread_group_start(group);
      thread_group_wait(group);
      num_threads_added = 0;
      aligned_stack = aligned_stack_start;
    }
  }

  if (num_threads_added > 0) {
    // Dispatch any tasks left in the group
    thread_group_start(group);
    thread_group_wait(group);
  }

  thread_group_free(group);

  return kTfLiteOk;
}

#else  // not XCORE

TfLiteStatus GenericDispatcher::Invoke(void** arguments, size_t size) const {
  std::vector<std::thread> group;

  for (int i = 0; i < size; i++) {
    std::thread th(function_, static_cast<void*>(arguments[i]));
    group.push_back(std::move(th));

    if (group.size() == num_threads_) {
      for (auto& thread : group) {
        thread.join();
      }
      group.clear();
    }
  }

  // Dispatch any tasks left in the group
  for (auto& thread : group) {
    thread.join();
  }

  return kTfLiteOk;
}

#endif

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
