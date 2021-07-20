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

#ifndef XCORE_PROFILER_H_
#define XCORE_PROFILER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_profiler.h"

#if !defined(XCORE_PROFILER_DEFAULT_MAX_LEVELS)
#define XCORE_PROFILER_DEFAULT_MAX_LEVELS (64)
#endif

namespace tflite {
namespace micro {
namespace xcore {

class XCoreProfiler : public tflite::MicroProfiler {
 public:
  explicit XCoreProfiler(){};
  ~XCoreProfiler() override = default;

  void Init(tflite::MicroAllocator* allocator,
            size_t max_event_count = XCORE_PROFILER_DEFAULT_MAX_LEVELS);

  void ClearEvents();

  uint32_t BeginEvent(const char* tag) override;

  // Event_handle is ignored since TFLu does not support concurrent events.
  void EndEvent(uint32_t event_handle) override;

  uint32_t const* GetEventDurations();
  size_t GetNumEvents();

 private:
  const char* event_tag_;
  uint32_t event_start_time_;
  size_t event_count_ = 0;
  size_t max_event_count_ = 0;
  uint32_t* event_durations_;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_PROFILER_H_
