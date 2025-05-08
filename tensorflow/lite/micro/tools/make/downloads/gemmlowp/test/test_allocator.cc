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
#include "../internal/allocator.h"

namespace gemmlowp {

void test_allocator(Allocator* a, int max_array_size) {
  const std::size_t int32_array_size = Random() % max_array_size;
  auto handle_to_int32_array = a->Reserve<std::int32_t>(int32_array_size);
  const std::size_t int8_array_size = Random() % max_array_size;
  auto handle_to_int8_array = a->Reserve<std::int8_t>(int8_array_size);
  a->Commit();
  std::int32_t* int32_array =
      a->GetPointer<std::int32_t>(handle_to_int32_array);
  std::int8_t* int8_array = a->GetPointer<std::int8_t>(handle_to_int8_array);
  Check(int32_array == a->GetPointer<std::int32_t>(handle_to_int32_array));
  Check(int8_array == a->GetPointer<std::int8_t>(handle_to_int8_array));
  Check(
      !(reinterpret_cast<std::uintptr_t>(int32_array) % Allocator::kAlignment));
  Check(
      !(reinterpret_cast<std::uintptr_t>(int8_array) % Allocator::kAlignment));
  Check(reinterpret_cast<std::uintptr_t>(int8_array) >=
        reinterpret_cast<std::uintptr_t>(int32_array + int32_array_size));
  memset(int32_array, 0, sizeof(*int32_array) * int32_array_size);
  memset(int8_array, 0, sizeof(*int8_array) * int8_array_size);
  a->Decommit();
}

void test_allocator() {
  Allocator allocator;

  // Test allocating increasingly large sizes on the same allocator,
  // starting with size 0.
  for (int i = 1; i < 1000; i += 10) {
    test_allocator(&allocator, i);
  }
}

}  // namespace gemmlowp

int main() { gemmlowp::test_allocator(); }
