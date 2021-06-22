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
#include "tensorflow/lite/micro/micro_profiler.h"

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

uint32_t MicroProfiler::BeginEvent(const char* tag) {
  if (num_events_ == kMaxEvents) {
    num_events_ = 0;
  }

  Event& event = events_[num_events_];
  int32_t start_tick = GetCurrentTimeTicks();
  event.id = num_events_;
  event.tag = tag;
  event.start_tick = start_tick;
  event.end_tick = start_tick - 1;
  return num_events_++;
}

void MicroProfiler::EndEvent(uint32_t event_handle) {
  TFLITE_DCHECK(event_handle < kMaxEvents);
  events_[event_handle].end_tick = GetCurrentTimeTicks();
}

int32_t MicroProfiler::GetTotalTicks() const {
  int32_t ticks = 0;
  for (int i = 0; i < num_events_; ++i) {
    ticks += events_[i].end_tick - events_[i].start_tick;
  }
  return ticks;
}

void MicroProfiler::Log() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  for (int i = 0; i < num_events_; ++i) {
    const Event& event = events_[i];
    int32_t ticks = event.end_tick - event.start_tick;
    MicroPrintf("%s took %d ticks (%d ms).", event.tag, ticks,
                TicksToMs(ticks));
  }
#endif
}

uint32_t MicroProfiler::GetTotalEvents() const { return num_events_; }

void MicroProfiler::GetActiveEvents(Event* events) const {
  memcpy(events, events_, sizeof(Event) * num_events_);
}

}  // namespace tflite
