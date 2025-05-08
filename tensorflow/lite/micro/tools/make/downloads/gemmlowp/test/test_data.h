// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
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

#ifndef GEMMLOWP_TEST_TEST_DATA_H_
#define GEMMLOWP_TEST_TEST_DATA_H_

namespace test_data {

extern const bool is_a_transposed;
extern const bool is_b_transposed;
extern const bool is_c_transposed;
extern const int m;
extern const int n;
extern const int k;
extern const int a_offset;
extern const int b_offset;
extern const int c_shift;
extern const int c_mult_int;
extern const int c_shift;
extern const int c_offset;

extern const int a_count;
extern const int b_count;
extern const int c_count;

extern unsigned char a_data[];
extern unsigned char b_data[];
extern unsigned char expected_c_data[];

}  // namespace test_data

#endif  // GEMMLOWP_TEST_TEST_DATA_H
