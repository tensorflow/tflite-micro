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

#include "tensorflow/lite/micro/static_vector.h"

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

using tflite::StaticVector;

TEST(StaticVectorTest, StaticVectorPushBack) {
  StaticVector<int, 4> a;
  EXPECT_EQ(a.max_size(), 4u);
  EXPECT_EQ(a.size(), 0u);

  a.push_back(1);
  EXPECT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0], 1);

  a.push_back(2);
  EXPECT_EQ(a.size(), 2u);
  EXPECT_EQ(a[1], 2);

  a.push_back(3);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[2], 3);
}

TEST(StaticVectorTest, StaticVectorInitializationPartial) {
  const StaticVector<int, 4> a{1, 2, 3};
  EXPECT_EQ(a.max_size(), 4u);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(a[2], 3);
}

TEST(StaticVectorTest, StaticVectorInitializationFull) {
  const StaticVector b{1, 2, 3};
  EXPECT_EQ(b.max_size(), 3u);
  EXPECT_EQ(b.size(), 3UL);
}

TEST(StaticVectorTest, StaticVectorEquality) {
  const StaticVector a{1, 2, 3};
  const StaticVector b{1, 2, 3};
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
}

TEST(StaticVectorTest, StaticVectorInequality) {
  const StaticVector a{1, 2, 3};
  const StaticVector b{3, 2, 1};
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
}

TEST(StaticVectorTest, StaticVectorSizeInequality) {
  const StaticVector a{1, 2};
  const StaticVector b{1, 2, 3};
  EXPECT_TRUE(a != b);
}

TEST(StaticVectorTest, StaticVectorPartialSizeInequality) {
  const StaticVector<int, 3> a{1, 2};
  const StaticVector<int, 3> b{1, 2, 3};
  EXPECT_TRUE(a != b);
}

TF_LITE_MICRO_TESTS_MAIN
