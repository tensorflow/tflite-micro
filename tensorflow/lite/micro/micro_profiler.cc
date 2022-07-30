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

#include <cinttypes>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

uint32_t MicroProfiler::BeginEvent(const char* tag) {
  if (num_events_ == kMaxEvents) {
    num_events_ = 0;
  }

  tags_[num_events_] = tag;
  start_ticks_[num_events_] = GetCurrentTimeTicks();
  end_ticks_[num_events_] = start_ticks_[num_events_] - 1;
  return num_events_++;
}

void MicroProfiler::EndEvent(uint32_t event_handle) {
  TFLITE_DCHECK(event_handle < kMaxEvents);
  end_ticks_[event_handle] = GetCurrentTimeTicks();
}

uint32_t MicroProfiler::GetTotalTicks() const {
  int32_t ticks = 0;
  for (int i = 0; i < num_events_; ++i) {
    ticks += end_ticks_[i] - start_ticks_[i];
  }
  return ticks;
}

void MicroProfiler::Log() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  for (int i = 0; i < num_events_; ++i) {
    uint32_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%s took %" PRIu32 " ticks (%d ms).", tags_[i], ticks,
                TicksToMs(ticks));
  }
#endif
}

void MicroProfiler::LogCsv() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  MicroPrintf("\"Event\",\"Tag\",\"Ticks\"");
  for (int i = 0; i < num_events_; ++i) {
    uint32_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%d,%s,%" PRIu32, i, tags_[i], ticks);
  }
#endif
}

void MicroProfiler::LogTicksPerTagCsv() {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  MicroPrintf(
      "\"Unique Ops in the Graph\",\"Total Ticks (all instances of the Op)\"");
  int totalTicks = 0;
  for (int i = 0; i < num_events_; ++i) {
    uint32_t ticks = end_ticks_[i] - start_ticks_[i];
    int position = FindExistingOrNextPosition(tags_[i]);
    total_ticks_per_tag[position].tag = tags_[i];
    total_ticks_per_tag[position].ticks =
        total_ticks_per_tag[position].ticks + ticks;
    totalTicks += ticks;
  }

  for (int i = 0; i < num_events_; ++i) {
    ticks_per_tag ticksPerTag = total_ticks_per_tag[i];
    if (ticksPerTag.tag == nullptr) {
      break;
    }
    MicroPrintf("%s,%d", ticksPerTag.tag, ticksPerTag.ticks);
  }
  MicroPrintf("total,%d", totalTicks);

#endif
}

int MicroProfiler::FindExistingOrNextPosition(const char* tagName) {
  int pos = 0;
  for (; pos < num_events_; pos++) {
    ticks_per_tag ticksPerTag = total_ticks_per_tag[pos];
    if (ticksPerTag.tag == nullptr) {
      return pos;
    } else {
      const char* currentTagName_t = ticksPerTag.tag;
      const char* newTagName_t = tagName;
      bool matched = true;
      while ((*currentTagName_t != '\0') && (*newTagName_t != '\0')) {
        if (((*currentTagName_t) == (*newTagName_t))) {
          ++currentTagName_t;
          ++newTagName_t;
        } else {
          matched = false;
          break;
        }
      }

      if (matched) {
        return pos;
      }
    }
  }
  return pos;
}

}  // namespace tflite
