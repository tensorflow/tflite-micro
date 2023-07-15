/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/tools/benchmarking/log_utils.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

namespace tflite {

int GetLongestStringLength(const char strings[][kMaxStringLength],
                           const int count) {
  int max_length = 0;

  for (int i = 0; i < count; ++i) {
    int size = strlen(strings[i]);
    if (size > max_length) {
      max_length = size;
    }
  }

  return max_length;
}

void FillColumnPadding(char* string, const int size, const int max_size,
                       const int padding) {
  FillString(string, max_size - size + padding, kMaxStringLength);
}

void FillString(char* string, const int size, const int buffer_size,
                const char value) {
  if (buffer_size <= (static_cast<int>(strlen(string)))) {
    for (int i = 0; i < buffer_size; ++i) {
      string[i] = (i < size) ? value : 0;
    }
  }
}

void MicroStrcat(char* output, const char* input, const int size) {
  if (size < 0) {
    strcat(output, input);  // NOLINT: strcat required due to no dynamic memory.
  } else {
    strncat(output, input, size);
  }
}

void MicroStrcpy(char* output, const char* input) {
  strcpy(output, input);  // NOLINT: strcpy required due to no dynamic memory.
}

void FormatIntegerDivide(char* output, const int64_t numerator,
                         const int64_t denominator, const int decimal_places) {
  int64_t multiplier = 1;
  for (int i = 0; i < decimal_places; ++i) {
    multiplier *= 10;
  }

  const int64_t total = numerator * multiplier / denominator;
  const int whole = static_cast<int>(total / multiplier);
  const int fractional = static_cast<int>(total % multiplier);
  sprintf(output, "%d.%d", whole, fractional);  // NOLINT: sprintf is required.
}

void FormatAsPercentage(char* output, const int64_t numerator,
                        const int64_t denominator, const int decimal_places) {
  FormatIntegerDivide(output, numerator * 100, denominator, decimal_places);
}

void PrettyPrintTableHeader(PrettyPrintType type, const char* table_name) {
  switch (type) {
    case PrettyPrintType::kCsv:
      MicroPrintf("[[ CSV ]]: %s", table_name);
      break;
    case PrettyPrintType::kTable:
      MicroPrintf("[[ TABLE ]]: %s", table_name);
  }
}

template <>
void FormatNumber<int32_t>(char* output, int32_t value) {
  sprintf(output, "%" PRId32, value);  // NOLINT: sprintf required.
}

template <>
void FormatNumber<size_t>(char* output, size_t value) {
  sprintf(output, "%zu", value);  // NOLINT: sprintf required.
}

template <>
void FormatNumber<float>(char* output, float value) {
  constexpr int64_t kDenominator = 1000;
  FormatIntegerDivide(output, static_cast<int64_t>(value * kDenominator),
                      kDenominator, 3);
}

template <>
void FormatNumber<double>(char* output, double value) {
  constexpr int64_t kDenominator = 1000;
  FormatIntegerDivide(output, static_cast<int64_t>(value * kDenominator),
                      kDenominator, 3);
}
}  // namespace tflite
