#ifndef TFLM_BENCHMARK_INTERNAL_METRICS_H_
#define TFLM_BENCHMARK_INTERNAL_METRICS_H_

#include <stdio.h>

#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/tools/benchmarking/internal/log_utils.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace tflm {
namespace benchmark {

// Logs the allocation events. Prints out two tables, one for the arena
// allocations, and one for each type of TFLM allocation type.
// Args:
//   - allocator: The recording micro allocator used during the invocation
//       process.
//   - type: Which print format should be used to output the allocation data to
//       stdout.
void LogAllocatorEvents(const tflite::RecordingMicroAllocator& allocator,
                        PrettyPrintType type);
}  // namespace benchmark
}  // namespace tflm

#endif  // TFLM_BENCHMARK_INTERNAL_METRICS_H_
