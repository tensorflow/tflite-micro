// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/lite/micro/span.h"

#include <array>

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestArrayInitialization) {
  int a[]{1, 2, 3};
  tflite::Span<int> s{a};
  TF_LITE_MICRO_EXPECT(s.data() == a);
  TF_LITE_MICRO_EXPECT(s.size() == sizeof(a) / sizeof(int));
}

TF_LITE_MICRO_TEST(TestStdArrayInitialization) {
  std::array<char, 20> a;
  tflite::Span<char> s{a};
  TF_LITE_MICRO_EXPECT(s.data() == a.data());
  TF_LITE_MICRO_EXPECT(s.size() == a.size());
}

TF_LITE_MICRO_TEST(TestEquality) {
  constexpr int a[]{1, 2, 3};
  constexpr int b[]{1, 2, 3};
  constexpr int c[]{3, 2, 1};
  tflite::Span<const int> s_a{a};
  tflite::Span<const int> s_b{b};
  tflite::Span<const int> s_c{c};
  TF_LITE_MICRO_EXPECT_TRUE(s_a == s_b);
  TF_LITE_MICRO_EXPECT_FALSE(s_a == s_c);
}

TF_LITE_MICRO_TEST(TestInequality) {
  constexpr int a[]{1, 2, 3};
  constexpr int b[]{1, 2, 3};
  constexpr int c[]{3, 2, 1};
  tflite::Span<const int> s_a{a};
  tflite::Span<const int> s_b{b};
  tflite::Span<const int> s_c{c};
  TF_LITE_MICRO_EXPECT_FALSE(s_a != s_b);
  TF_LITE_MICRO_EXPECT_TRUE(s_a != s_c);
}

TF_LITE_MICRO_TESTS_END
