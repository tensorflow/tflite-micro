/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TFLM_BENCHMARK_INTERNAL_MICRO_BENCHMARK_H_
#define TFLM_BENCHMARK_INTERNAL_MICRO_BENCHMARK_H_

#include <climits>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"

namespace tflm {
namespace benchmark {

// TODO(b/256914173): Remove kTfLiteAbort once all models have migrated off of
// using CircularBuffer op.
constexpr TfLiteStatus kTfLiteAbort = static_cast<TfLiteStatus>(-9);

template <typename inputT>
class MicroBenchmarkRunner {
 public:
  // The lifetimes of model, op_resolver, tensor_arena, profiler must exceed
  // that of the created MicroBenchmarkRunner object.
  MicroBenchmarkRunner(const tflite::Model* model,
                       const tflite::MicroOpResolver* op_resolver,
                       uint8_t* tensor_arena, int tensor_arena_size,
                       tflite::MicroProfilerInterface* profiler)
      : interpreter_(CreateInterpreter(model, op_resolver, tensor_arena,
                                       tensor_arena_size, profiler)) {}

  TfLiteStatus AllocateTensors(tflite::MicroProfilerInterface* profiler) {
    tflite::ScopedMicroProfiler scoped("AllocateTensors", profiler);
    return interpreter_.AllocateTensors();
  }

  TfLiteStatus RunSingleIteration() { return interpreter_.Invoke(); }

  void SetRandomInput(const int random_seed) {
    // The pseudo-random number generator is initialized to a constant seed
    std::srand(random_seed);
    for (size_t i = 0; i < interpreter_.inputs_size(); ++i) {
      TfLiteTensor* input = interpreter_.input_tensor(i);

      // Pre-populate input tensor with random values.
      int input_length = input->bytes / sizeof(inputT);
      inputT* input_values = tflite::GetTensorData<inputT>(input);
      for (int j = 0; j < input_length; j++) {
        // Pre-populate input tensor with a random value based on a constant
        // seed.
        input_values[j] = static_cast<inputT>(
            std::rand() % (std::numeric_limits<inputT>::max() -
                           std::numeric_limits<inputT>::min() + 1));
      }
    }
  }

  void SetInput(const inputT* custom_input, const size_t index) {
    TfLiteTensor* input = interpreter_.input_tensor(index);
    if (input == nullptr) {
      MicroPrintf("Cannot set input data for invalid input tensor index: %zu",
                  index);
      std::abort();
    }
    inputT* input_buffer = tflite::GetTensorData<inputT>(input);
    memcpy(input_buffer, custom_input, input->bytes);
  }

  TfLiteTensor* GetOutput(const int index) {
    return interpreter_.output(index);
  }

  const tflite::RecordingMicroAllocator& GetMicroAllocator() const {
    return interpreter_.GetMicroAllocator();
  }

 private:
  static tflite::RecordingMicroInterpreter CreateInterpreter(
      const tflite::Model* model, const tflite::MicroOpResolver* op_resolver,
      uint8_t* tensor_arena, int tensor_arena_size,
      tflite::MicroProfilerInterface* profiler) {
    tflite::ScopedMicroProfiler scoped("CreateInterpreter", profiler);
    tflite::MicroAllocator* allocator =
        tflite::MicroAllocator::Create(tensor_arena, tensor_arena_size);
    return tflite::RecordingMicroInterpreter(
        model, *op_resolver, tensor_arena, tensor_arena_size,
        tflite::MicroResourceVariables::Create(allocator, 24), profiler);
  }

  tflite::RecordingMicroInterpreter interpreter_;
};
}  // namespace benchmark
}  // namespace tflm

#endif  // TFLM_BENCHMARK_INTERNAL_MICRO_BENCHMARK_H_
