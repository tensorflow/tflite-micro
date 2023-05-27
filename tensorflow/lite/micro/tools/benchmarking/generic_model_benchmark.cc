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

#include <sys/stat.h>
#include <sys/types.h>

#include <memory>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/tools/benchmarking/log_utils.h"
#include "tensorflow/lite/micro/tools/benchmarking/metrics.h"
#include "tensorflow/lite/micro/tools/benchmarking/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/*
 * Generic model benchmark.  Evaluates runtime performance of a provided model
 * with random inputs.
 */

namespace tflm {
namespace benchmark {

namespace {

using Profiler = ::tflite::MicroProfiler;

using TflmOpResolver = tflite::MicroMutableOpResolver<96>;

// TODO(b/256914173): Remove kTfLiteAbort once all models have migrated off of
// using CircularBuffer op.
constexpr TfLiteStatus kTfLiteAbort = static_cast<TfLiteStatus>(-9);

// Seed used for the random input. Input data shouldn't affect invocation timing
// so randomness isn't really needed.
constexpr int kRandomSeed = 0xF742BE52;

// Which format should be used to output debug information.
constexpr PrettyPrintType kPrintType = PrettyPrintType::kCsv;

constexpr int kTensorArenaSize = 1024 * 1024;
constexpr int kNumResourceVariable = 100;
constexpr int kModelSize = 511408;

void SetRandomInput(const int random_seed,
                    tflite::RecordingMicroInterpreter& interpreter) {
  // The pseudo-random number generator is initialized to a constant seed
  std::srand(random_seed);
  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input_tensor(i);

    // Pre-populate input tensor with random values.
    int input_length = input->bytes / sizeof(int8_t);
    int8_t* input_values = tflite::GetTensorData<int8_t>(input);
    for (int j = 0; j < input_length; j++) {
      // Pre-populate input tensor with a random value based on a constant
      // seed.
      input_values[j] = static_cast<int8_t>(
          std::rand() % (std::numeric_limits<int8_t>::max() -
                         std::numeric_limits<int8_t>::min() + 1));
    }
  }
}

int ReadFile(const char* file_name, char* buffer) {
  // Obtain the file size using fstat, or report an error if that fails.
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(file_name, "rb"), fclose);
  struct stat sb;

  if (fstat(fileno(file.get()), &sb) != 0) {
    MicroPrintf("Failed to get file size of: %s\n", file_name);
    return -1;
  }

  size_t buffer_size_bytes_ = sb.st_size;

  if (!buffer) {
    MicroPrintf("Malloc of buffer to hold copy of '%s' failed\n", file_name);
    return -1;
  }

  size_t bytes_read =
      fread((void*)buffer, sizeof(char), buffer_size_bytes_, file.get());

  if (bytes_read > kModelSize) {
    MicroPrintf(
        "Buffer size (%d) to hold the model is less than required %d.\n",
        kModelSize, bytes_read);
  }

  return 0;
}

int Benchmark(const char* model_file_name) {
  Profiler profiler;
  // Create an area of memory to use for input, output, and intermediate arrays.
  // Align arena to 16 bytes to avoid alignment warnings on certain platforms.
  // Tensor size is dependent on the core being used since some of them have
  // limited memory available. The TENSOR_ARENA_SIZE macro is defined in the
  // build rules.
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  char model_file_content[kModelSize];

  ReadFile(model_file_name, model_file_content);
  uint32_t event_handle = profiler.BeginEvent("TfliteGetModel");
  const tflite::Model* model = tflite::GetModel(model_file_content);
  profiler.EndEvent(event_handle);

  TflmOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(tflm::benchmark::CreateOpResolver(op_resolver));

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      model, op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariable),
      &profiler);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  PrettyPrintTableHeader(kPrintType, "Initialization");
  profiler.LogCsv();
  profiler.ClearEvents();

  MicroPrintf("");  // null MicroPrintf serves as a newline.

  // For streaming models, the interpreter will return kTfLiteAbort if the model
  // does not yet have enough data to make an inference. As such, we need to
  // invoke the interpreter multiple times until we either receive an error or
  // kTfLiteOk. This loop also works for non-streaming models, as they'll just
  // return kTfLiteOk after the first invocation.
  int seed = kRandomSeed;
  while (true) {
    tflm::benchmark::SetRandomInput(seed++, interpreter);
    TfLiteStatus status = interpreter.Invoke();
    if ((status != kTfLiteOk) && (status != kTfLiteAbort)) {
      MicroPrintf("Model interpreter invocation failed: %d\n", status);
      return -1;
    }

    PrettyPrintTableHeader(kPrintType, "Invocation");
    profiler.LogCsv();
    MicroPrintf("");  // null MicroPrintf serves as a newline.
    profiler.ClearEvents();

    if (status == kTfLiteOk) {
      break;
    }
  }

  LogAllocatorEvents(*allocator, kPrintType);

  return 0;
}
}  // namespace
}  // namespace benchmark
}  // namespace tflm

int main(int argc, char** argv) { return tflm::benchmark::Benchmark(argv[1]); }
