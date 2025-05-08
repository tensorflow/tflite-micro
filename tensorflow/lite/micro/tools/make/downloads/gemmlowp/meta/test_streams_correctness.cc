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

#include <unistd.h>
#ifdef __APPLE__
#include <sys/time.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "streams.h"

#define MUL_OFFSET (3)
#define ADD_OFFSET (100)

using namespace gemmlowp::meta;

void prepare_row_major_data(int rows, int elements, int stride, std::uint8_t* data) {
  for (int i = 0; i < rows * stride; ++i) {
    data[i] = 255;
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < elements; ++j) {
      data[i * stride + j] = j % 256;
    }
  }
}

void prepare_column_major_data(int columns, int elements, int stride,
                               std::uint8_t* data) {
  for (int i = 0; i < elements * stride; ++i) {
    data[i] = 255;
  }
  for (int i = 0; i < elements; ++i) {
    for (int j = 0; j < columns; ++j) {
      data[i * stride + j] = i % 256;
    }
  }
}

void print_out(std::uint8_t* result, int rows, int elements) {
  int size = rows * ((elements + 7) / 8) * 8;
  for (int i = 0; i < size; ++i) {
    std::cout << static_cast<int>(result[i]) << " ";
  }
  std::cout << std::endl << std::flush;
}

bool check(std::uint8_t* result, int rows, int elements) {
  int chunks = elements / 8;
  int leftover = elements % 8;
  for (int i = 0; i < chunks; ++i) {
    int chunk_index = i * rows * 8;
    int chunk_start_value = i * 8;
    for (int j = 0; j < rows; ++j) {
      for (int k = 0; k < 8; ++k) {
        if (result[chunk_index + j * 8 + k] != chunk_start_value + k) {
          return false;
        }
      }
    }
  }

  int leftover_index = chunks * rows * 8;
  int leftover_start_value = chunks * 8;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < leftover; ++j) {
      if (result[leftover_index + i * 8 + j] != leftover_start_value + j) {
        return false;
      }
    }
  }

  int expected_sum =
      ((elements * (elements - 1)) / 2) * MUL_OFFSET + ADD_OFFSET;
  int sums_offset = rows * ((elements + 7) / 8) * 8;
  std::int32_t* sums = reinterpret_cast<std::int32_t*>(result + sums_offset);
  for (int i = 0; i < rows; ++i) {
    if (sums[i] != expected_sum) {
      return false;
    }
  }

  return true;
}

template <int lanes, int leftover>
void test_2(std::uint8_t* in, std::uint8_t* out) {
  for (int elements = 8; elements < 64; elements += 8) {
    int all_elements = elements + leftover;
    for (int stride = all_elements; stride < all_elements + 4; ++stride) {
      RowMajorWithSum params;
      params.count = all_elements;
      params.stride = stride;
      params.multiplicative_sum_offset = MUL_OFFSET;
      params.additive_sum_offset = ADD_OFFSET;

      prepare_row_major_data(lanes, all_elements, stride, in);
      Stream<std::uint8_t, lanes, 8, leftover, RowMajorWithSum>::Pack(in, params,
                                                                 out);
      if (check(out, lanes, all_elements)) {
        //        std::cout << "Row: " << lanes << "x8x" << leftover << " : "
        //                  << all_elements << "@" << stride << " -- OK" <<
        //                  std::endl;
      } else {
        std::cout << "Row: " << lanes << "x8x" << leftover << " : "
                  << all_elements << "@" << stride << " -- ERROR" << std::endl;
        std::cout << "Exiting." << std::endl;
        std::exit(1);
      }
    }

    for (int stride = lanes; stride < lanes + 4; ++stride) {
      ColumnMajorWithSum params;
      params.count = all_elements;
      params.stride = stride;
      params.multiplicative_sum_offset = MUL_OFFSET;
      params.additive_sum_offset = ADD_OFFSET;

      prepare_column_major_data(lanes, all_elements, stride, in);
      Stream<std::uint8_t, lanes, 8, leftover, ColumnMajorWithSum>::Pack(in, params,
                                                                    out);
      if (check(out, lanes, all_elements)) {
        //        std::cout << "Column: " << lanes << "x8x" << leftover << " : "
        //                  << all_elements << "@" << stride << " -- OK" <<
        //                  std::endl;
      } else {
        std::cout << "Column: " << lanes << "x8x" << leftover << " : "
                  << all_elements << "@" << stride << " -- ERROR" << std::endl;
        std::cout << "Exiting." << std::endl;
        std::exit(1);
      }
    }
  }
}

template <int lanes>
void test(std::uint8_t* in, std::uint8_t* out) {
  test_2<lanes, 0>(in, out);
  test_2<lanes, 1>(in, out);
  test_2<lanes, 2>(in, out);
  test_2<lanes, 3>(in, out);
  test_2<lanes, 4>(in, out);
  test_2<lanes, 5>(in, out);
  test_2<lanes, 6>(in, out);
  test_2<lanes, 7>(in, out);
}

int main() {
  std::unique_ptr<std::uint8_t> in(new std::uint8_t[128 * 1024]);
  std::unique_ptr<std::uint8_t> out(new std::uint8_t[128 * 1024]);

  test<1>(in.get(), out.get());
  test<2>(in.get(), out.get());
  test<3>(in.get(), out.get());
  test<4>(in.get(), out.get());
  test<5>(in.get(), out.get());
  test<6>(in.get(), out.get());
  test<7>(in.get(), out.get());
  test<8>(in.get(), out.get());

  std::cout << "Ok." << std::endl;
  return 0;
}
