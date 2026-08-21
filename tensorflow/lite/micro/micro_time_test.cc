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

#include "tensorflow/lite/micro/micro_time.h"

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(MicroTimeTest, TestBasicTimerFunctionality) {
  const uint32_t ticks_per_second = tflite::ticks_per_second();

  // If the platform does not implement a timer, skip the test.
  if (ticks_per_second == 0) {
    return;
  }

  const auto start_time = tflite::GetCurrentTimeTicks();

  // HARDENING: Increase retries to handle fast CPUs on platforms with low
  // timer resolution (e.g., Windows ~15ms). 100 million iterations guarantees
  // the loop exceeds the duration of a single tick.
  constexpr int kMaxRetries = 100000000;

  for (volatile int i = 0; i < kMaxRetries; i++) {
    if (tflite::GetCurrentTimeTicks() != start_time) {
      break;
    }
  }

  // Verify that time has actually advanced.
  EXPECT_GT(tflite::GetCurrentTimeTicks(), start_time);
}

TF_LITE_MICRO_TESTS_MAIN
