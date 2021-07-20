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

#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {
namespace micro {
namespace xcore {

void XCoreProfiler::Init(tflite::MicroAllocator* allocator,
                         size_t max_event_count) {
  max_event_count_ = max_event_count;
  event_durations_ = static_cast<uint32_t*>(
      allocator->AllocatePersistentBuffer(max_event_count * sizeof(uint32_t)));
}

uint32_t const* XCoreProfiler::GetEventDurations() { return event_durations_; }

size_t XCoreProfiler::GetNumEvents() { return event_count_; }

void XCoreProfiler::ClearEvents() { event_count_ = 0; }

uint32_t XCoreProfiler::BeginEvent(const char* tag) {
  TFLITE_DCHECK(tag);
  event_tag_ = tag;
  event_start_time_ = tflite::GetCurrentTimeTicks();
  return 0;
}

void XCoreProfiler::EndEvent(uint32_t event_handle) {
  int32_t event_end_time = tflite::GetCurrentTimeTicks();
  event_count_ = event_count_ % max_event_count_;
  // wrap if there are too many events
  event_durations_[event_count_++] = event_end_time - event_start_time_;
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
