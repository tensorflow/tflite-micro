// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>  // NOLINT
#include <vector>

#include "../internal/multi_thread_gemm.h"
#include "../profiling/pthread_everywhere.h"
#include "test.h"

namespace gemmlowp {

class Thread {
 public:
  Thread(BlockingCounter* blocking_counter, int number_of_times_to_decrement)
      : blocking_counter_(blocking_counter),
        number_of_times_to_decrement_(number_of_times_to_decrement),
        made_the_last_decrement_(false),
        finished_(false) {
    pthread_create(&thread_, nullptr, ThreadFunc, this);
  }

  ~Thread() { Join(); }

  bool Join() {
    while (!finished_.load()) {
    }
    return made_the_last_decrement_;
  }

 private:
  Thread(const Thread& other) = delete;

  void ThreadFunc() {
    for (int i = 0; i < number_of_times_to_decrement_; i++) {
      Check(!made_the_last_decrement_);
      made_the_last_decrement_ = blocking_counter_->DecrementCount();
    }
    finished_.store(true);
  }

  static void* ThreadFunc(void* ptr) {
    static_cast<Thread*>(ptr)->ThreadFunc();
    return nullptr;
  }

  BlockingCounter* const blocking_counter_;
  const int number_of_times_to_decrement_;
  pthread_t thread_;
  bool made_the_last_decrement_;
  // finished_ is used to manually implement Join() by busy-waiting.
  // I wanted to use pthread_join / std::thread::join, but the behavior
  // observed on Android was that pthread_join aborts when the thread has
  // already joined before calling pthread_join, making that hard to use.
  // It appeared simplest to just implement this simple spinlock, and that
  // is good enough as this is just a test.
  std::atomic<bool> finished_;
};

void test_blocking_counter(BlockingCounter* blocking_counter, int num_threads,
                           int num_decrements_per_thread,
                           int num_decrements_to_wait_for) {
  std::vector<Thread*> threads;
  blocking_counter->Reset(num_decrements_to_wait_for);
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(new Thread(blocking_counter, num_decrements_per_thread));
  }
  blocking_counter->Wait();

  int num_threads_that_made_the_last_decrement = 0;
  for (int i = 0; i < num_threads; i++) {
    if (threads[i]->Join()) {
      num_threads_that_made_the_last_decrement++;
    }
    delete threads[i];
  }
  Check(num_threads_that_made_the_last_decrement == 1);
}

void test_blocking_counter() {
  BlockingCounter* blocking_counter = new BlockingCounter;

  // repeating the entire test sequence ensures that we test
  // non-monotonic changes.
  for (int repeat = 1; repeat <= 2; repeat++) {
    for (int num_threads = 1; num_threads <= 5; num_threads++) {
      for (int num_decrements_per_thread = 1;
           num_decrements_per_thread <= 4 * 1024;
           num_decrements_per_thread *= 16) {
        test_blocking_counter(blocking_counter, num_threads,
                              num_decrements_per_thread,
                              num_threads * num_decrements_per_thread);
      }
    }
  }
  delete blocking_counter;
}

}  // end namespace gemmlowp

int main() { gemmlowp::test_blocking_counter(); }
