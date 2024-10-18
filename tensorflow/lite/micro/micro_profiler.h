/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

enum class MicroProfilerLogFormat {
  HumanReadable,
  Csv,
};

// MicroProfiler creates a common way to gain fine-grained insight into runtime
// performance. Bottleneck operators can be identified along with slow code
// sections. This can be used in conjunction with running the relevant micro
// benchmark to evaluate end-to-end performance.
template <int MAX_EVENTS>
class MicroProfiler : public MicroProfilerInterface {
 public:
  MicroProfiler() = default;
  virtual ~MicroProfiler() = default;

  // Marks the start of a new event and returns an event handle that can be used
  // to mark the end of the event via EndEvent. The lifetime of the tag
  // parameter must exceed that of the MicroProfiler.
  uint32_t BeginEvent(const char* tag) {
    if (num_events_ == MAX_EVENTS) {
      MicroPrintf(
          "MicroProfiler errored out because total number of events exceeded "
          "the maximum of %d.",
          MAX_EVENTS);
      TFLITE_ASSERT_FALSE;
    }

    tags_[num_events_] = tag;
    start_ticks_[num_events_] = GetCurrentTimeTicks();
    end_ticks_[num_events_] = start_ticks_[num_events_] - 1;
    return num_events_++;
  }

  // Marks the end of an event associated with event_handle. It is the
  // responsibility of the caller to ensure than EndEvent is called once and
  // only once per event_handle.
  //
  // If EndEvent is called more than once for the same event_handle, the last
  // call will be used as the end of event marker.If EndEvent is called 0 times
  // for a particular event_handle, the duration of that event will be 0 ticks.
  void EndEvent(uint32_t event_handle) {
    TFLITE_DCHECK(event_handle < MAX_EVENTS);
    end_ticks_[event_handle] = GetCurrentTimeTicks();
  }

  // Clears all the events that have been currently profiled.
  void ClearEvents() {
    num_events_ = 0;
    num_tag_groups_ = 0;
  }

  // Returns the sum of the ticks taken across all the events. This number
  // is only meaningful if all of the events are disjoint (the end time of
  // event[i] <= start time of event[i+1]).
  uint32_t GetTotalTicks() const {
    int32_t ticks = 0;
    for (int i = 0; i < num_events_; ++i) {
      ticks += end_ticks_[i] - start_ticks_[i];
    }
    return ticks;
  }

  // Prints the profiling information of each of the events in human readable
  // form.
  void Log(MicroProfilerLogFormat format) const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    switch (format) {
      case MicroProfilerLogFormat::HumanReadable:
        for (int i = 0; i < num_events_; ++i) {
          uint32_t ticks = end_ticks_[i] - start_ticks_[i];
          uint64_t us = TicksToUs(ticks);
          MicroPrintf("%s took %u.%u ms (%u ticks)", tags_[i], us / 1000,
                      us % 1000, ticks);
        }
        break;

      case MicroProfilerLogFormat::Csv:
        MicroPrintf("\"Event\",\"Tag\",\"Ms\",\"Ticks\"");
        for (int i = 0; i < num_events_; ++i) {
#if defined(HEXAGON) || defined(CMSIS_NN)
          int ticks = end_ticks_[i] - start_ticks_[i];
          MicroPrintf("%d,%s,%u,%d", i, tags_[i], TicksToMs(ticks), ticks);
#else
          uint32_t ticks = end_ticks_[i] - start_ticks_[i];
          MicroPrintf("%d,%s,%" PRIu32 ",%" PRIu32, i, tags_[i],
                      TicksToMs(ticks), ticks);
#endif
        }
        break;
    }
#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)
  }

  // Prints the profiling information of each of the events in human readable
  // form, grouped per tag, sorted by execution time.
  void LogGrouped(MicroProfilerLogFormat format) {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    for (int i = 0; i < num_events_; ++i) {
      // Find if the tag already exists in uniqueTags
      TagGroup& tag_group = GetTagGroup(tags_[i]);

      uint32_t ticks = end_ticks_[i] - start_ticks_[i];
      tag_group.tag = tags_[i];
      tag_group.ticks += ticks;
      tag_group.tag_count++;
    }

    SortTagGroups();

    switch (format) {
      case MicroProfilerLogFormat::HumanReadable: {
        MicroPrintf("Cumulative event times:");
        MicroPrintf("%-8s %-32s %-12s %-12s", "Count", "Tag", "Ticks", "Time");
        uint64_t total_ticks = 0;
        uint64_t us;
        for (int i = 0; i < num_tag_groups_; ++i) {
          total_ticks += tag_groups_[i].ticks;
          us = TicksToUs(tag_groups_[i].ticks);
          MicroPrintf("%-8d %-32s %-12d %" PRIu64 ".%03" PRIu64 " ms",
                      tag_groups_[i].tag_count, tag_groups_[i].tag,
                      tag_groups_[i].ticks, us / 1000, us % 1000);
        }
        us = TicksToUs(total_ticks);
        MicroPrintf("\nTotal time: %" PRIu64 ".%03" PRIu64 " ms (%lld ticks)",
                    us / 1000, us % 1000, total_ticks);
        break;
      }
      case MicroProfilerLogFormat::Csv: {
        MicroPrintf("\"Tag\",\"Total ms\",\"Total ticks\"");
        for (int i = 0; i < num_tag_groups_; ++i) {
          MicroPrintf("%s, %u, %u", tag_groups_[i].tag,
                      TicksToMs(tag_groups_[i].ticks), tag_groups_[i].ticks);
        }
        break;
      }
    }
#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)
  }

 private:
  const char* tags_[MAX_EVENTS];
  uint32_t start_ticks_[MAX_EVENTS];
  uint32_t end_ticks_[MAX_EVENTS];
  int num_events_ = 0;
  int num_tag_groups_ = 0;

  struct TagGroup {
    const char* tag;
    uint32_t ticks;
    uint32_t tag_count;
  };

  // In practice, the number of tags will be much lower than the number of
  // events. But it is theoretically possible that each event to be unique and
  // hence we allow total_ticks_per_tag to have MAX_EVENTS entries.
  TagGroup tag_groups_[MAX_EVENTS] = {};

  // Helper function to find the index of a tag in the cumulative array
  TagGroup& GetTagGroup(const char* tag) {
    for (int i = 0; i < num_tag_groups_; ++i) {
      if (strcmp(tag_groups_[i].tag, tag) == 0) {
        return tag_groups_[i];
      }
    }

    // Tag not found, so we create a new entry
    // There should always be space since the array of tag groups
    // is just as big as the array of events
    tag_groups_[num_tag_groups_].tag = tag;
    tag_groups_[num_tag_groups_].ticks = 0;
    tag_groups_[num_tag_groups_].tag_count = 0;
    return tag_groups_[num_tag_groups_++];
  }

  // Helper function to sort the tag groups by ticks in descending order
  // Simple bubble sort implementation
  void SortTagGroups() {
    for (int i = 0; i < num_tag_groups_ - 1; ++i) {
      for (int j = i + 1; j < num_tag_groups_; ++j) {
        if (tag_groups_[j].ticks > tag_groups_[i].ticks) {
          TagGroup temp_tag_group = tag_groups_[i];
          tag_groups_[i] = tag_groups_[j];
          tag_groups_[j] = temp_tag_group;
        }
      }
    }
  }

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

#if defined(TF_LITE_STRIP_ERROR_STRINGS)
// For release builds, the ScopedMicroProfiler is a noop.
//
// This is done because the ScopedProfiler is used as part of the
// MicroInterpreter and we want to ensure zero overhead for the release builds.
class ScopedMicroProfiler {
 public:
  explicit ScopedMicroProfiler(const char* tag,
                               MicroProfilerInterface* profiler) {}
};

#else

// This class can be used to add events to a MicroProfiler object that span the
// lifetime of the ScopedMicroProfiler object.
// Usage example:
//
// MicroProfiler profiler();
// ...
// {
//   ScopedMicroProfiler scoped_profiler("custom_tag", profiler);
//   work_to_profile();
// }
class ScopedMicroProfiler {
 public:
  explicit ScopedMicroProfiler(const char* tag,
                               MicroProfilerInterface* profiler)
      : profiler_(profiler) {
    if (profiler_ != nullptr) {
      event_handle_ = profiler_->BeginEvent(tag);
    }
  }

  ~ScopedMicroProfiler() {
    if (profiler_ != nullptr) {
      profiler_->EndEvent(event_handle_);
    }
  }

 private:
  uint32_t event_handle_ = 0;
  MicroProfilerInterface* profiler_ = nullptr;
};
#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_
