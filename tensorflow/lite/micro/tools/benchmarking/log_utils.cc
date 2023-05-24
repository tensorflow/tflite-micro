#include "tensorflow/lite/micro/tools/benchmarking/log_utils.h"

#include <stdio.h>
#include <stdlib.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/micro_log.h"

namespace tflm {
namespace benchmark {

int GetMaxStringLength(const char strings[][kMaxStringLength],
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
  for (int i = 0; i < buffer_size; ++i) {
    string[i] = (i < size) ? value : 0;
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
      break;
    case PrettyPrintType::kBase64:
      MicroPrintf("[[ BASE64 ]]: %s", table_name);
  }
}

void Base64Encode(const uint8_t* input, char* output, const size_t input_len,
                  const size_t output_len) {
  if (!CheckEncodedBase64Length(input_len, output_len)) {
    MicroPrintf(
        "Output buffer (%d bytes) is too small to support input data (%d "
        "bytes).",
        output_len, input_len);
    abort();
  }

  static const char* b64 =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  unsigned int i, j, a, b, c;
  // For every 3 bytes of input data, write 4 bytes of base64.
  for (i = j = 0; i < input_len; i += 3) {
    a = input[i];
    b = i + 1 >= input_len ? 0 : input[i + 1];
    c = i + 2 >= input_len ? 0 : input[i + 2];

    // Take the first 6 bits.
    // [0 1 2 3 4 5 6 7] -> [0 1 2 3 4 5]
    output[j++] = b64[a >> 2];

    // Take the last 2 bits of a, and the first 4 of b.
    // [0 1 2 3 4 5 6 7, 8 9 10 11 12 13 14 15] -> [6 7 8 9 10 11].
    output[j++] = b64[((a & 0x3) << 4) | (b >> 4)];
    if (i + 1 < input_len) {
      // Take the last 4 bits of b, and the first 2 of c.
      // [8 9 10 11 12 13 14 15, 16 17 18 19 20 21 22 23] -> [12 13 14 15 16 17]
      output[j++] = b64[(b & 0xF) << 2 | (c >> 6)];
    }
    if (i + 2 < input_len) {
      // Take the last 6 bits of c.
      // [16 17 18 19 20 21 22 23] -> [18 19 20 21 22 23]
      output[j++] = b64[c & 0x3F];
    }
  }

  // Add padding.
  while (j % 4 != 0) {
    output[j++] = '=';
  }

  output[j++] = '\0';
}

bool CheckEncodedBase64Length(const size_t input_len, const size_t output_len) {
  const float required_len = static_cast<float>(4.0 / 3.0 * input_len + 4);
  return static_cast<float>(output_len) >= required_len;
}

template <>
void FormatNumber<int32_t>(char* output, int32_t value) {
  sprintf(output, "%d", value);  // NOLINT: sprintf required.
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
}  // namespace benchmark
}  // namespace tflm
