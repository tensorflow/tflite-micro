/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {

NonPersistentMemoryPlannerShim::NonPersistentMemoryPlannerShim(
    const BufferPlan* offline_buffer_plan)
    : offline_buffer_plan_(offline_buffer_plan) {}

NonPersistentMemoryPlannerShim::~NonPersistentMemoryPlannerShim() {}

TfLiteStatus NonPersistentMemoryPlannerShim::AddBuffer(
    tflite::ErrorReporter* error_reporter, int size, int first_time_used,
    int last_time_used) {
  MicroPrintf("Unsupported operation");
  return kTfLiteError;
}

size_t NonPersistentMemoryPlannerShim::GetMaximumMemorySize() {
  // Simply return 0 to let the framework accept this memory plan
  // because the client ensure validity of the memory plan.
  return 0;
}

// How many buffers are in the given memory plan.
int NonPersistentMemoryPlannerShim::GetBufferCount() {
  return offline_buffer_plan_->buffer_count;
}

TfLiteStatus NonPersistentMemoryPlannerShim::GetOffsetForBuffer(
    ErrorReporter* error_reporter, int buffer_index, int* offset) {
  if (buffer_index >= offline_buffer_plan_->buffer_count) {
    MicroPrintf("buffer index %d is outside range 0 to %d", buffer_index,
                offline_buffer_plan_->buffer_count);
    return kTfLiteError;
  }
  *offset = offline_buffer_plan_->buffer_plan_entries[buffer_index].offset;
  return kTfLiteOk;
}

}  // namespace tflite
