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

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>
#include <memory>
#include <random>
#include <type_traits>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/tools/benchmarking/metrics.h"
#include "tensorflow/lite/micro/tools/benchmarking/op_resolver.h"
#include "tensorflow/lite/micro/tools/benchmarking/show_meta_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)
#if !defined(GENERIC_BENCHMARK_MODEL_HEADER_PATH)
#error "GENERIC_BENCHMARK_MODEL_HEADER_PATH missing from CXXFLAGS"
#endif  // !defined(GENERIC_BENCHMARK_MODEL_HEADER_PATH)
#if !defined(GENERIC_BENCHMARK_MODEL_NAME)
#error "GENERIC_BENCHMARK_MODEL_NAME missing from CXXFLAGS"
#endif  // !defined(GENERIC_BENCHMARK_MODEL_NAME)

#include GENERIC_BENCHMARK_MODEL_HEADER_PATH

#define __MODEL_DATA(x) g_##x##_model_data
#define _MODEL_DATA(x) __MODEL_DATA(x)
#define MODEL_DATA _MODEL_DATA(GENERIC_BENCHMARK_MODEL_NAME)
#define __MODEL_SIZE(x) g_##x##_model_data_size
#define _MODEL_SIZE(x) __MODEL_SIZE(x)
#define MODEL_SIZE _MODEL_SIZE(GENERIC_BENCHMARK_MODEL_NAME)

#endif  // defind(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)

/*
 * Generic model benchmark.  Evaluates runtime performance of a provided model
 * with random inputs.
 */

namespace tflite {

namespace {

using Profiler = ::tflite::MicroProfiler;

// Seed used for the random input. Input data shouldn't affect invocation timing
// so randomness isn't really needed.
constexpr uint32_t kRandomSeed = 0xFB;

#if !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)
constexpr size_t kTensorArenaSize = 3e6;
constexpr size_t kModelSize = 2e6;
#elif defined(GENERIC_BENCHMARK_TENSOR_ARENA_SIZE)
constexpr size_t kTensorArenaSize = GENERIC_BENCHMARK_TENSOR_ARENA_SIZE;
#else
constexpr size_t kTensorArenaSize = 5e6 - MODEL_SIZE;
#endif  // !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)

constexpr int kNumResourceVariable = 100;

void SetRandomInput(const uint32_t random_seed,
                    tflite::MicroInterpreter& interpreter) {
  std::mt19937 eng(random_seed);
  std::uniform_int_distribution<uint32_t> dist(0, 255);

  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input_tensor(i);

    // Pre-populate input tensor with random values.
    int8_t* input_values = tflite::GetTensorData<int8_t>(input);
    for (size_t j = 0; j < input->bytes; ++j) {
      input_values[j] = dist(eng);
    }
  }
}

#if !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)

struct FileCloser {
  void operator()(FILE* file) { fclose(file); }
};

bool ReadFile(const char* file_name, void* buffer, size_t buffer_size) {
  std::unique_ptr<FILE, FileCloser> file(fopen(file_name, "rb"));

  const size_t bytes_read =
      fread(buffer, sizeof(char), buffer_size, file.get());
  if (ferror(file.get())) {
    MicroPrintf("Unable to read model file: %d\n", ferror(file.get()));
    return false;
  }
  if (!feof(file.get())) {
    // Note that http://b/297592546 can mean that this error message is
    // confusing.
    MicroPrintf(
        "Model buffer (%d bytes) is too small for the model (%d bytes).\n",
        buffer_size, bytes_read);
    return false;
  }
  if (bytes_read == 0) {
    MicroPrintf("No bytes read from model file.\n");
    return false;
  }

  return true;
}
#endif  // !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)

int Benchmark(const uint8_t* model_data, tflite::PrettyPrintType print_type) {
  Profiler profiler;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  uint32_t event_handle = profiler.BeginEvent("TfliteGetModel");
  const tflite::Model* model = tflite::GetModel(model_data);
  profiler.EndEvent(event_handle);

  TflmOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(CreateOpResolver(op_resolver));

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      model, op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariable),
      &profiler);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  profiler.Log();
  profiler.ClearEvents();

  MicroPrintf("");  // null MicroPrintf serves as a newline.

  // For streaming models, the interpreter will return kTfLiteAbort if the model
  // does not yet have enough data to make an inference. As such, we need to
  // invoke the interpreter multiple times until we either receive an error or
  // kTfLiteOk. This loop also works for non-streaming models, as they'll just
  // return kTfLiteOk after the first invocation.
  uint32_t seed = kRandomSeed;
  while (true) {
    SetRandomInput(seed++, interpreter);
    TfLiteStatus status = interpreter.Invoke();
    if ((status != kTfLiteOk) && (static_cast<int>(status) != kTfLiteAbort)) {
      MicroPrintf("Model interpreter invocation failed: %d\n", status);
      return -1;
    }

    profiler.Log();
    MicroPrintf("");  // null MicroPrintf serves as a newline.
    profiler.LogTicksPerTagCsv();
    MicroPrintf("");  // null MicroPrintf serves as a newline.
    profiler.ClearEvents();

    if (status == kTfLiteOk) {
      break;
    }
  }

  LogAllocatorEvents(*allocator, print_type);

  return 0;
}
}  // namespace
}  // namespace tflite

#if !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)
void usage(const char* prog_name) {
  MicroPrintf("usage: %s filename [--csv]", prog_name);
}
#endif  // !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)

int main(int argc, char** argv) {
  // Which format should be used to output debug information.
  tflite::PrettyPrintType print_type = tflite::PrettyPrintType::kTable;
  tflite::InitializeTarget();

#if !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)
  if (argc < 2 || argc > 3) {
    usage(argv[0]);
    return -1;
  }
  const char* model_filename = argv[1];

  if (argc == 3) {
    if (std::strcmp(argv[2], "--csv") == 0) {
      print_type = tflite::PrettyPrintType::kCsv;
    } else {
      usage(argv[0]);
      return -1;
    }
  }

  alignas(16) static uint8_t model_data[tflite::kModelSize];

  if (!tflite::ReadFile(model_filename, model_data, tflite::kModelSize)) {
    return -1;
  }
#else
  const uint8_t* model_data = MODEL_DATA;
#endif  // !defined(GENERIC_BENCHMARK_USING_BUILTIN_MODEL)

  MicroPrintf("\nConfigured arena size = %d\n", tflite::kTensorArenaSize);
  tflite::GenericBenchmarkShowMetaData();
  return tflite::Benchmark(model_data, print_type);
}
