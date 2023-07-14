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

#ifndef TFLM_BENCHMARK_INTERNAL_LOG_UTILS_H_
#define TFLM_BENCHMARK_INTERNAL_LOG_UTILS_H_

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

// The maxmimum length of a string.
static constexpr int kMaxStringLength = 32;

// The maximum length of a table row, applies to the header as well.
static constexpr int kMaxRowLength = 100;

// The default padding between columns in a table.
static constexpr int kDefaultColumnPadding = 4;

// Defines how formatted data is printed to stdout.
enum class PrettyPrintType {
  // Prints as a CSV file.
  kCsv,
  // Prints as a formatted table.
  kTable,
};

// Returns the length of the longest string in an array.
// Args:
// - strings: An array of strings.
// - count: The number of strings in the array.
int GetLongestStringLength(const char strings[][kMaxStringLength], int count);

// Adds padding between two columns in a table.
// ex) "hello" is being inserted into a column. The largest value in that column
//     is 10, and there's a global padding of 4 spaces. Therefore, 9 spaces (10
//     - 5 + 4) are added as padding.
// Args:
// - string: The input padding string.
// - size: The size of the string that's being inserted into a column.
// - max_size: The size of the largest string in the column.
// - padding: The amount of padding to add to each column regardless of its
//     size.
void FillColumnPadding(char* string, int size, int max_size,
                       int padding = kDefaultColumnPadding);

// Fills a string with a specified value.
// Args:
// - string: The input string. This is filled in with the specified value.
// - size: The size of the string after being filled in. This must be less than
//     the allocated space for the string.
// - buffer_size: The size of the string's buffer.
// - value: The value to insert into the string. Defaults to a space.
void FillString(char* string, int size, int buffer_size, char value = ' ');

// Concatenates the input string onto the first.
// Args:
// - output: The destination string for where to append input.
// - input: The input string to concatenate.
// - size: The number of characters to concatenate from the first string. If
//     negative, the whole input string will be concatenated.
void MicroStrcat(char* output, const char* input, int size = -1);

// Copies the input string into the output.
void MicroStrcpy(char* output, const char* input);

// Formats a division operation to have a specified number of decimal places.
// Args:
// - output: The output string to be formatted.
// - numerator: The numerator in the division operation.
// - denominator: The denominator in the division operation.
// - decimal places: The number of decimal places to print to.
void FormatIntegerDivide(char* output, int64_t numerator, int64_t denominator,
                         int decimal_places);

// Formats a division operation as a percentage.
// Args:
// - output: The output string to be formatted.
// - numerator: The numerator in the division operation.
// - denominator: The denominator in the division operation.
// - decimal places: The number of decimal places to print to.
void FormatAsPercentage(char* output, int64_t numerator, int64_t denominator,
                        int decimal_places);

void PrettyPrintTableHeader(PrettyPrintType type, const char* table_name);

// Formats a number as a string.
// Args:
// - output: The location of where to write the formatted number.
// - value: The value to write to a string.
template <typename T>
void FormatNumber(char* output, T value);

// Pretty prints a table to stdout.
// Note: kMaxRows and kColumns should describe the allocated size of the table,
//       not the amount of data that is populated. It is required that all
//       columns are filled out, but not all rows.
//
// ex) PrintTable<3, 25>(headers, data, 4);
//     This will print a table with 3 columns and 4 rows. In this example, it
//     is required that data is defined as char[3][25][kMaxStringLength] to
//     properly print.
//
// op        cycles    cpu %
// -------------------------
// foo     | 1000     | 10
// bar     | 2500     | 25
// baz     | 1000     | 10
// lorem   | 2000     | 20
//
// Args:
// - headers: A 1D array of strings containing the headers of the table. This
//     must be equal in size to kColumns.
// - data: A 2D array of string data organized in [columns, rows]. As stated
//     above, it is required that all columns are populated, but not all rows.
// - rows: The number of populated rows in `data`.
template <int kMaxRows, int kColumns>
void PrintTable(const char headers[kColumns][kMaxStringLength],
                const char data[kColumns][kMaxRows][kMaxStringLength],
                const int rows) {
  // Get the maximum width for each column in the table.
  int max_column_width[kColumns];
  for (int i = 0; i < kColumns; ++i) {
    max_column_width[i] = std::max(GetLongestStringLength(data[i], rows),
                                   static_cast<int>(strlen(headers[i])));
  }

  // Add padding between each item in the header so it can be printed on one
  // line.
  char header_spaces[kColumns][kMaxStringLength];
  for (int i = 0; i < kColumns; ++i) {
    FillColumnPadding(header_spaces[i], strlen(headers[i]), max_column_width[i],
                      kDefaultColumnPadding + 2);
  }

  // Print the header.
  char header[kMaxRowLength];
  memset(header, 0, kMaxRowLength);
  for (int i = 0; i < kColumns; ++i) {
    MicroStrcat(header, headers[i]);
    MicroStrcat(header, header_spaces[i]);
  }
  MicroPrintf("%s", header);

  // Print a separator to separate the header from the data.
  char separator[kMaxRowLength];
  FillString(separator, strlen(header) - 1, kMaxRowLength, '-');
  MicroPrintf("%s", separator);

  for (int i = 0; i < rows; ++i) {
    char spaces[kColumns][kMaxStringLength];
    for (int j = 0; j < kColumns; ++j) {
      FillColumnPadding(spaces[j], strlen(data[j][i]), max_column_width[j]);
    }

    char row[kMaxRowLength];
    memset(row, 0, kMaxRowLength);

    // Concatenate each column in a row with the format "[data][padding]| "
    for (int j = 0; j < kColumns; ++j) {
      MicroStrcat(row, data[j][i]);
      MicroStrcat(row, spaces[j]);
      MicroStrcat(row, "| ");
    }

    MicroPrintf("%s", row);
  }

  MicroPrintf(separator);
  MicroPrintf("");
}

// Pretty prints a csv to stdout.
// Note: kMaxRows and kColumns should describe the allocated size of the table,
//       not the amount of data that is populated. It is required that all
//       columns are filled out, but not all rows.
//
// ex)
// op,cycles,%cpu
// foo,1000,10
// bar,2500,25
// baz,1000,10
//
// Args:
// - headers: A 1D array of strings containing the headers of the table. This
//     must be equal in size to kColumns.
// - data: A 2D array of string data organized in [columns, rows]. As stated
//     above, it is required that all columns are populated, but not all rows.
// - rows: The number of populated rows in `data`.
template <int kMaxRows, int kColumns>
void PrintCsv(const char headers[kColumns][kMaxStringLength],
              const char data[kColumns][kMaxRows][kMaxStringLength],
              const int rows) {
  char header[kMaxRowLength];
  memset(header, 0, kMaxRowLength);
  for (int i = 0; i < kColumns; ++i) {
    MicroStrcat(header, headers[i]);
    if (i < kColumns - 1) {
      MicroStrcat(header, ",");
    }
  }

  MicroPrintf("%s", header);

  char row[kMaxRowLength];
  for (int i = 0; i < rows; ++i) {
    memset(row, 0, kMaxRowLength);
    for (int j = 0; j < kColumns; ++j) {
      MicroStrcat(row, data[j][i]);
      if (j < kColumns - 1) {
        MicroStrcat(row, ",");
      }
    }

    MicroPrintf("%s", row);
  }

  MicroPrintf("");  // Serves as a new line.
}

// Prints a 2D array of strings in a formatted manner along with a table name
// that includes the table type.
//
// Note: kMaxRows and kColumns should describe the allocated size of the table,
//       not the amount of data that is populated. It is required that all
//       columns are filled out, but not all rows.
//
// ex) PrettyPrint::kCsv will print a csv with a [[ CSV ]]: table_name header.
//
// Args:
// - headers: A 1D array of strings containing the headers of the table. This
//     must be equal in size to kColumns.
// - data: A 2D array of string data organized in [columns, rows]. As stated
//     above, it is required that all columns are populated, but not all rows.
// - rows: The number of populated rows in `data`.
// - type: The format type that should be used to pretty print.
// - table_name: The name of the table to be printed alongside the format type.
template <int kMaxRows, int kColumns>
void PrintFormattedData(const char headers[kColumns][kMaxStringLength],
                        const char data[kColumns][kMaxRows][kMaxStringLength],
                        const int rows, const PrettyPrintType type,
                        const char* table_name) {
  PrettyPrintTableHeader(type, table_name);
  switch (type) {
    case PrettyPrintType::kCsv:
      PrintCsv<kMaxRows, kColumns>(headers, data, rows);
      break;
    case PrettyPrintType::kTable:
      PrintTable<kMaxRows, kColumns>(headers, data, rows);
      break;
  }
}

}  // namespace tflite

#endif  // TFLM_BENCHMARK_INTERNAL_LOG_UTILS_H_
