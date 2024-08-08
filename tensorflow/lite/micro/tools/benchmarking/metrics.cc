/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/tools/benchmarking/metrics.h"

#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

struct LogArenaRecord {
  const char* title;
  int allocations;
  float percentage;
};

struct LogAllocationRecord {
  const char* title;
  int type;
  int used_bytes;
  int requested_bytes;
  int count;
  float percentage;
};

constexpr int kArenaRows = 3;
constexpr int kArenaColumns = 3;

constexpr int kAllocationTypes =
    static_cast<int>(tflite::RecordedAllocationType::kNumAllocationTypes);
constexpr int kAllocationColumns = 6;

constexpr int kMaxBufSize = 100;

LogArenaRecord GetLogArenaRecord(
    const tflite::RecordingSingleArenaBufferAllocator* allocator,
    int row_index) {
  TFLITE_DCHECK(row_index < kArenaRows);

  const size_t total_bytes = allocator->GetUsedBytes();
  const size_t allocations[] = {total_bytes,
                                allocator->GetNonPersistentUsedBytes(),
                                allocator->GetPersistentUsedBytes()};
  static_assert(std::extent<decltype(allocations)>::value == kArenaRows,
                "kArenaRows mismatch");
  const char* titles[] = {"Total", "NonPersistent", "Persistent"};
  static_assert(std::extent<decltype(titles)>::value == kArenaRows,
                "kArenaRows mismatch");

  LogArenaRecord record = {};
  record.title = titles[row_index];
  record.allocations = allocations[row_index];
  record.percentage = record.allocations * 100.0f / total_bytes;

  return record;
}

LogAllocationRecord GetLogAllocationRecord(
    const tflite::RecordingMicroAllocator& allocator, int row_index) {
  TFLITE_DCHECK(row_index < kAllocationTypes);

  const tflite::RecordedAllocationType types[] = {
      tflite::RecordedAllocationType::kTfLiteEvalTensorData,
      tflite::RecordedAllocationType::kPersistentTfLiteTensorData,
      tflite::RecordedAllocationType::kPersistentTfLiteTensorQuantizationData,
      tflite::RecordedAllocationType::kPersistentBufferData,
      tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData,
      tflite::RecordedAllocationType::kNodeAndRegistrationArray,
      tflite::RecordedAllocationType::kOpData,
#ifdef USE_TFLM_COMPRESSION
      tflite::RecordedAllocationType::kCompressionData,
#endif  // USE_TFLM_COMPRESSION
  };
  static_assert(std::extent<decltype(types)>::value == kAllocationTypes,
                "kAllocationTypes mismatch");
  const char* titles[] = {
      "Eval tensor data",
      "Persistent tensor data",
      "Persistent quantization data",
      "Persistent buffer data",
      "Tensor variable buffer data",
      "Node and registration array",
      "Operation data",
#ifdef USE_TFLM_COMPRESSION
      "Compression data",
#endif  // USE_TFLM_COMPRESSION
  };
  static_assert(std::extent<decltype(titles)>::value == kAllocationTypes,
                "kAllocationTypes mismatch");
  const size_t total_bytes =
      allocator.GetSimpleMemoryAllocator()->GetUsedBytes();
  tflite::RecordedAllocation allocation =
      allocator.GetRecordedAllocation(types[row_index]);

  LogAllocationRecord record = {};
  record.title = titles[row_index];
  record.type = static_cast<int>(types[row_index]);
  record.used_bytes = allocation.used_bytes;
  record.requested_bytes = allocation.requested_bytes;
  record.count = allocation.count;
  record.percentage = allocation.used_bytes * 100.0f / total_bytes;

  return record;
}

template <int kColumns>
void UpdateColumnWidths(int (&widths)[kColumns], const char* s[kColumns]) {
  for (int i = 0; i < kColumns; i++) {
    widths[i] = std::max(widths[i], static_cast<int>(std::strlen(s[i])));
  }
}

void UpdateColumnWidths(int (&widths)[kArenaColumns],
                        const LogArenaRecord& record) {
  char buf[kMaxBufSize];
  int count;

  count = MicroSnprintf(buf, kMaxBufSize, "%s", record.title);
  widths[0] = std::max(widths[0], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%d", record.allocations);
  widths[1] = std::max(widths[1], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%.2f",
                        static_cast<double>(record.percentage));
  widths[2] = std::max(widths[2], count);
}

void UpdateColumnWidths(int (&widths)[kAllocationColumns],
                        const LogAllocationRecord& record) {
  char buf[kMaxBufSize];
  int count;

  count = MicroSnprintf(buf, kMaxBufSize, "%s", record.title);
  widths[0] = std::max(widths[0], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%d", record.type);
  widths[1] = std::max(widths[1], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%d", record.used_bytes);
  widths[2] = std::max(widths[2], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%d", record.requested_bytes);
  widths[3] = std::max(widths[3], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%d", record.count);
  widths[4] = std::max(widths[4], count);
  count = MicroSnprintf(buf, kMaxBufSize, "%.2f",
                        static_cast<double>(record.percentage));
  widths[5] = std::max(widths[5], count);
}

using BufferDatum = std::tuple<char*, char*>;

template <typename T>
BufferDatum AddTableColumnValue(const BufferDatum& buffer, const char* format,
                                int column_width, T value,
                                const char* separator = nullptr) {
  char* p;
  char* p_end;
  std::tie(p, p_end) = buffer;
  int count = MicroSnprintf(p, p_end - p, format, column_width, value);
  p += count;
  if (separator != nullptr && p < p_end) {
    count = MicroSnprintf(p, p_end - p, separator);
    p += count;
  }

  if (p > p_end) {
    p = p_end;
  }

  return std::make_tuple(p, p_end);
}

}  // namespace

void LogArenaAllocations(
    const tflite::RecordingSingleArenaBufferAllocator* allocator,
    const PrettyPrintType type) {
  const char* headers[] = {"Arena", "Bytes", "%% Arena"};
  static_assert(std::extent<decltype(headers)>::value == kArenaColumns,
                "kArenaColumns mismatch");
  char buffer[kMaxBufSize];
  BufferDatum buffer_datum =
      std::make_tuple(std::begin(buffer), std::end(buffer));
  int column_widths[kArenaColumns] = {};

  const char* output_type;
  const char* string_format;
  if (type == PrettyPrintType::kCsv) {
    output_type = "CSV";
    string_format = "\"%*s\"";
  } else {
    output_type = "Table";
    string_format = "%*s";

    UpdateColumnWidths<kArenaColumns>(column_widths, headers);
    for (int i = 0; i < kArenaRows; i++) {
      LogArenaRecord record = GetLogArenaRecord(allocator, i);
      UpdateColumnWidths(column_widths, record);
    }
  }

  MicroPrintf("[[ %s ]]: Arena", output_type);

  for (int i = 0; i < kArenaColumns; i++) {
    // create header
    const char* separator = nullptr;
    if (i != kArenaColumns - 1) {
      // separator for all but last column value
      if (type == PrettyPrintType::kCsv) {
        separator = ",";
      } else {
        separator = "   ";
      }
    }
    buffer_datum = AddTableColumnValue(buffer_datum, string_format,
                                       column_widths[i], headers[i], separator);
  }
  MicroPrintf(buffer);

  for (int i = 0; i < kArenaRows; ++i) {
    // create rows
    const char* separator = (type == PrettyPrintType::kCsv) ? "," : " | ";
    buffer_datum = std::make_tuple(std::begin(buffer), std::end(buffer));
    LogArenaRecord record = GetLogArenaRecord(allocator, i);
    buffer_datum = AddTableColumnValue(
        buffer_datum, string_format, column_widths[0], record.title, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*d", column_widths[1],
                                       record.allocations, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*.2f", column_widths[2],
                                       static_cast<double>(record.percentage));
    MicroPrintf(buffer);
  }

  MicroPrintf("");  // output newline
}

void LogAllocations(const tflite::RecordingMicroAllocator& allocator,
                    const PrettyPrintType type) {
  const char* headers[] = {"Allocation", "Id",    "Used",
                           "Requested",  "Count", "%% Memory"};
  static_assert(std::extent<decltype(headers)>::value == kAllocationColumns,
                "kAllocationColumns mismatch");
  char buffer[kMaxBufSize];
  BufferDatum buffer_datum =
      std::make_tuple(std::begin(buffer), std::end(buffer));
  int column_widths[kAllocationColumns] = {};

  const char* output_type;
  const char* string_format;
  if (type == PrettyPrintType::kCsv) {
    output_type = "CSV";
    string_format = "\"%*s\"";
  } else {
    output_type = "Table";
    string_format = "%*s";

    UpdateColumnWidths<kAllocationColumns>(column_widths, headers);
    for (int i = 0; i < kAllocationTypes; i++) {
      LogAllocationRecord record = GetLogAllocationRecord(allocator, i);
      UpdateColumnWidths(column_widths, record);
    }
  }

  MicroPrintf("[[ %s ]]: Allocations", output_type);

  for (int i = 0; i < kAllocationColumns; i++) {
    // create header
    const char* separator = nullptr;
    if (i != kAllocationColumns - 1) {
      // separator for all but last column value
      if (type == PrettyPrintType::kCsv) {
        separator = ",";
      } else {
        separator = "   ";
      }
    }
    buffer_datum = AddTableColumnValue(buffer_datum, string_format,
                                       column_widths[i], headers[i], separator);
  }
  MicroPrintf(buffer);

  for (int i = 0; i < kAllocationTypes; ++i) {
    // create rows
    const char* separator = (type == PrettyPrintType::kCsv) ? "," : " | ";
    buffer_datum = std::make_tuple(std::begin(buffer), std::end(buffer));
    LogAllocationRecord record = GetLogAllocationRecord(allocator, i);
    buffer_datum = AddTableColumnValue(
        buffer_datum, string_format, column_widths[0], record.title, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*d", column_widths[1],
                                       record.type, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*d", column_widths[2],
                                       record.used_bytes, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*d", column_widths[3],
                                       record.requested_bytes, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*d", column_widths[4],
                                       record.count, separator);
    buffer_datum = AddTableColumnValue(buffer_datum, "%*.2f", column_widths[5],
                                       static_cast<double>(record.percentage));
    MicroPrintf(buffer);
  }

  MicroPrintf("");  // output newline
}

void LogAllocatorEvents(const tflite::RecordingMicroAllocator& allocator,
                        const PrettyPrintType type) {
  LogArenaAllocations(allocator.GetSimpleMemoryAllocator(), type);
  LogAllocations(allocator, type);
}

}  // namespace tflite
