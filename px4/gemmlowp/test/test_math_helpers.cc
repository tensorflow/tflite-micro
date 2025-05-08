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

#include "test.h"

#include <limits>

#include "../internal/common.h"

namespace gemmlowp {

// Our math helpers don't intend to be reliable all the way to the
// limit of representable range, wrt overflow.
// We don't care for 2G sized matrices.
// This test stops at half of the representable range.
template <typename Integer>
Integer ValueRangeCutoff() {
  return std::numeric_limits<Integer>::max() / 2;
}

int RandomNonnegativeFarAwayFromOverflow() { return Random() % (1 << 24); }

template <int Modulus>
void test_round_up_down(int x) {
  Check(x >= RoundDown<Modulus>(x));
  Check(x < RoundDown<Modulus>(x) + Modulus);
  Check(RoundDown<Modulus>(x) % Modulus == 0);

  Check(x <= RoundUp<Modulus>(x));
  Check(x > RoundUp<Modulus>(x) - Modulus);
  Check(RoundUp<Modulus>(x) % Modulus == 0);
}

template <int Modulus>
void test_round_up_down() {
  for (int i = 0; i < 100; i++) {
    test_round_up_down<Modulus>(i);
    const int N = ValueRangeCutoff<int>();
    test_round_up_down<Modulus>(Random() % N);
  }
}

template <typename Integer>
void test_ceil_quotient(Integer x, Integer y) {
  Check(CeilQuotient(x, y) * y >= x);
  Check(CeilQuotient(x, y) * y < x + y);
}

template <typename Integer>
void test_ceil_quotient() {
  const Integer N = ValueRangeCutoff<Integer>();
  const Integer K = std::min(N, Integer(100));
  for (Integer x = 0; x < K; x++) {
    for (Integer y = 1; y < K; y++) {
      test_ceil_quotient(x, y);
      test_ceil_quotient(x, Integer(1 + (Random() % (N - 1))));
      test_ceil_quotient(Integer(Random() % N), y);
      test_ceil_quotient(Integer(Random() % N),
                         Integer(1 + (Random() % (N - 1))));
    }
  }
}

template <typename Integer>
void test_round_up_to_next_power_of_two(Integer x) {
  Check(RoundUpToPowerOfTwo(RoundUpToPowerOfTwo(x) == RoundUpToPowerOfTwo(x)));
  Check(RoundUpToPowerOfTwo(x) >= x);
  Check(x == 0 || RoundUpToPowerOfTwo(x) < 2 * x);
  Check((RoundUpToPowerOfTwo(x) & (RoundUpToPowerOfTwo(x) - 1)) == 0);
}

template <typename Integer>
void test_round_up_to_next_power_of_two() {
  const Integer N = ValueRangeCutoff<Integer>();
  const Integer K = std::min(N, Integer(100));
  for (Integer x = 0; x < K; x++) {
    test_round_up_to_next_power_of_two(x);
    test_round_up_to_next_power_of_two(Random() % N);
  }
}

void test_math_helpers() {
  test_round_up_down<1>();
  test_round_up_down<2>();
  test_round_up_down<3>();
  test_round_up_down<4>();
  test_round_up_down<5>();
  test_round_up_down<6>();
  test_round_up_down<7>();
  test_round_up_down<8>();
  test_round_up_down<9>();
  test_round_up_down<10>();
  test_round_up_down<11>();
  test_round_up_down<12>();
  test_round_up_down<13>();
  test_round_up_down<14>();
  test_round_up_down<15>();
  test_round_up_down<16>();

  test_round_up_down<50>();
  test_round_up_down<51>();

  test_round_up_down<500>();
  test_round_up_down<501>();

  test_ceil_quotient<std::int8_t>();
  test_ceil_quotient<std::uint8_t>();
  test_ceil_quotient<std::int16_t>();
  test_ceil_quotient<std::uint16_t>();
  test_ceil_quotient<std::int32_t>();
  test_ceil_quotient<std::uint32_t>();

  test_round_up_to_next_power_of_two<std::int8_t>();
  test_round_up_to_next_power_of_two<std::uint8_t>();
  test_round_up_to_next_power_of_two<std::int16_t>();
  test_round_up_to_next_power_of_two<std::uint16_t>();
  test_round_up_to_next_power_of_two<std::int32_t>();
  test_round_up_to_next_power_of_two<std::uint32_t>();
}

}  // end namespace gemmlowp

int main() { gemmlowp::test_math_helpers(); }
