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

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <memory>
#include <random>

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

namespace tflite {

namespace {

using Profiler = ::tflite::MicroProfiler;

using TflmOpResolver = tflite::MicroMutableOpResolver<96>;

constexpr int kTfLiteAbort = -9;

// Seed used for the random input. Input data shouldn't affect invocation timing
// so randomness isn't really needed.
constexpr uint32_t kRandomSeed = 0xFB;

// Which format should be used to output debug information.
constexpr PrettyPrintType kPrintType = PrettyPrintType::kTable;

constexpr size_t kTensorArenaSize = 3e6;
constexpr int kNumResourceVariable = 100;
constexpr size_t kModelSize = 2e6;

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

bool ReadFile(const char* file_name, void* buffer, size_t buffer_size) {
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(file_name, "rb"), fclose);

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

int Benchmark(const char* model_file_name) {
  Profiler profiler;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  alignas(16) static uint8_t model_file_content[kModelSize];

  if (!ReadFile(model_file_name, model_file_content, kModelSize)) {
    return -1;
  }
  uint32_t event_handle = profiler.BeginEvent("TfliteGetModel");
  const tflite::Model* model = tflite::GetModel(model_file_content);
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

  LogAllocatorEvents(*allocator, kPrintType);

  return 0;
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) { return tflite::Benchmark(argv[1]); }
