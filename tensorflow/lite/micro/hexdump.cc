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

#include "tensorflow/lite/micro/hexdump.h"

#include <iomanip>
#include <iostream>
#include <ostream>

#include "tensorflow/lite/micro/static_vector.h"

void tflite::hexdump(tflite::Span<const std::byte> region, std::ostream& out) {
  auto initial_flags = out.flags();
  out << std::hex << std::uppercase << std::setfill('0');

  std::size_t byte = 0;
  constexpr int per_line = 16;
  const int lines = (region.size() + per_line - 1) / per_line;  // rounded up
  for (int line = 0; line < lines; ++line) {
    tflite::StaticVector<char, per_line> ascii;

    // print address
    out << std::setw(8) << line << ":";

    for (int pos = 0; pos < per_line; ++pos) {
      if (byte < region.size()) {
        // print byte
        int as_int = static_cast<int>(region[byte++]);
        out << ' ' << std::setw(2) << as_int;

        // buffer an ascii printable value
        char c{'.'};
        if (std::isprint(as_int)) {
          c = static_cast<char>(as_int);
        }
        ascii.push_back(c);
      } else {
        out << "   ";
      }

      // extra space in the middle
      if (pos == per_line / 2 - 1) {
        out << " ";
      }
    }

    out << "  ";
    for (const auto& c : ascii) {
      out << c;
    }
    out << '\n';
  }

  out.flags(initial_flags);
}
