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

// Implementation for the DebugLog() function that prints to the debug logger on
// an generic Cortex-M device.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "tensorflow/lite/micro/debug_log.h"

#include <stdio.h>

#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

// Implements the debug log callback when using Semihosting.
// Semihosting uses the host os for system calls so using
// stdio seems appropriate here.
void SemiHostingCallback(const char* s) { printf("%s", s); }

#ifdef USE_SEMIHOSTING
// Semihosting will output to stdio of the host machine.
static DebugLogCallback debug_log_callback = (*SemiHostingCallback);
#else
// Otherwise use a stub.
static DebugLogCallback debug_log_callback = nullptr;
#endif

void RegisterDebugLogCallback(void (*cb)(const char* s)) {
  debug_log_callback = cb;
}

void DebugLog(const char* s) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  if (debug_log_callback != nullptr) {
    debug_log_callback(s);
  }
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
