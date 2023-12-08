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

#include "tensorflow/lite/micro/debug_log.h"

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include "eyalroz_printf/src/printf/printf.h"
#endif

namespace {

#ifndef TF_LITE_STRIP_ERROR_STRINGS
// For Arm Cortex-M devices, calling SYS_WRITE0 will output the zero-terminated
// string pointed to by R1 to any debug console that's attached to the system.
void SysWriteDebugConsole(const char* s) {
  asm("mov r0, #0x04\n"  // SYS_WRITE0
      "mov r1, %[str]\n"
      "bkpt #0xAB\n"
      :
      : [str] "r"(s)
      : "r0", "r1");
}
#endif  // TF_LITE_STRIP_ERROR_STRINGS

}  // namespace

extern "C" void DebugLog(const char* format, va_list args) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  constexpr int kMaxLogLen = 256;
  char log_buffer[kMaxLogLen];

  vsnprintf_(log_buffer, kMaxLogLen, format, args);
  SysWriteDebugConsole(log_buffer);
#endif  // TF_LITE_STRIP_ERROR_STRINGS
}

#ifndef TF_LITE_STRIP_ERROR_STRINGS
// Only called from MicroVsnprintf (micro_log.h)
extern "C" int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                              va_list vlist) {
  return vsnprintf_(buffer, buf_size, format, vlist);
}
#endif
