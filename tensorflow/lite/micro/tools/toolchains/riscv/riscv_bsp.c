/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>
#include <sys/time.h>
#include <sys/times.h>

// Provides _times() for bare-metal RV32IMC toolchains whose libgloss omits it
// (such as SiFive GCC 8.1), using semihosted gettimeofday() so ticks match
// Newlib's CLOCKS_PER_SEC = 1,000,000 (1 us) under QEMU and falling back to
// rdcycle on pure bare-metal targets.
__attribute__((weak)) clock_t _times(struct tms* buf) {
  static time_t base_sec = 0;
  static int initialized = 0;
  struct timeval tv;
  clock_t ticks;
  if (gettimeofday(&tv, 0) == 0) {
    if (!initialized) {
      base_sec = tv.tv_sec;
      initialized = 1;
    }
    uint64_t us =
        (uint64_t)(tv.tv_sec - base_sec) * 1000000ULL + (uint64_t)tv.tv_usec;
    ticks = (clock_t)us;
  } else {
    uint32_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    ticks = (clock_t)cycles;
  }
  if (ticks == (clock_t)-1) {
    ticks = 0;
  }
  if (buf != 0) {
    buf->tms_utime = ticks;
    buf->tms_stime = 0;
    buf->tms_cutime = 0;
    buf->tms_cstime = 0;
  }
  return ticks;
}
