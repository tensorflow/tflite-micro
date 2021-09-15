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

#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {
// Change this number to have a binary with a different
// size of data section.
constexpr int kSize = 64;
// Initialize this global array so that it goes to data section, not bss.
long random_array[kSize] = {1, 2, 3, 4, 5, 6, 7, 8};
}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(BinarySizeChangeBykSize) {
  // Just some code to create a binary and keep data section.
  for (int i = 0; i < kSize; i++) {
    random_array[i] = i + 1;
  }

  for (int i = 0; i < kSize; i++) {
    TF_LITE_MICRO_EXPECT_EQ(random_array[i], i + 1);
  }
}

TF_LITE_MICRO_TESTS_END
