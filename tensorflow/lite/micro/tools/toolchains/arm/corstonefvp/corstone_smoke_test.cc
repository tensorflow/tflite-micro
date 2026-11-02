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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#if defined(__ARM_FEATURE_MVE)
#include <arm_mve.h>
#endif

#include <ethosu_driver.h>

extern "C" uint64_t corstone_get_cycles(void);

namespace {

volatile int g_c_ctor_ran = 0;
volatile int g_cpp_ctor_ran = 0;

__attribute__((constructor)) void OnInitArray(void) { g_c_ctor_ran = 1; }

__attribute__((destructor)) void OnFiniArray(void) {
  printf("CORSTONE_FINI_ARRAY_OK\n");
}

struct GlobalInitProbe {
  GlobalInitProbe() { g_cpp_ctor_ran = 1; }
};

GlobalInitProbe g_probe;

}  // namespace

int main() {
  printf("Hello from Arm Corstone-300 + Ethos-U55 FVP!\n");

  if (g_c_ctor_ran != 1 || g_cpp_ctor_ran != 1) {
    printf("FAILURE: .init_array constructors did not run (%d, %d)\n",
           g_c_ctor_ran, g_cpp_ctor_ran);
    return 1;
  }

#if defined(__ARM_FEATURE_MVE)
  int32x4_t a = vdupq_n_s32(10);
  int32x4_t b = vdupq_n_s32(32);
  int32x4_t c = vaddq_s32(a, b);
  int lane0 = vgetq_lane_s32(c, 0);
  printf("Helium MVE: 10 + 32 = %d\n", lane0);
#else
  int lane0 = 10 + 32;
  printf("Scalar: 10 + 32 = %d\n", lane0);
#endif
  if (lane0 != 42) {
    printf("FAILURE: Unexpected computation result %d\n", lane0);
    return 1;
  }

  struct ethosu_driver* drv = ethosu_reserve_driver();
  if (drv == nullptr) {
    printf("FAILURE: Ethos-U55 NPU driver could not be reserved\n");
    return 1;
  }
  printf("Ethos-U55 NPU driver successfully reserved!\n");
  ethosu_release_driver(drv);

  // Verify dynamic _sbrk heap (>32 KiB) and ENOMEM bounds enforcement.
  constexpr size_t kOneMiB = 1024U * 1024U;
  auto* buf = static_cast<uint8_t*>(malloc(kOneMiB));
  __asm__ volatile("" : "+r"(buf) : : "memory");
  if (buf == nullptr) {
    printf("FAILURE: 1 MiB malloc returned nullptr\n");
    return 1;
  }
  buf[0] = 0x5AU;
  buf[kOneMiB - 1] = 0xA5U;
  __asm__ volatile("" : "+r"(buf) : : "memory");
  if (buf[0] != 0x5AU || buf[kOneMiB - 1] != 0xA5U) {
    printf("FAILURE: Heap memory readback mismatch\n");
    free(buf);
    return 1;
  }
  free(buf);

  errno = 0;
  constexpr size_t kEightMiB = 8U * 1024U * 1024U;
  void* too_big = malloc(kEightMiB);
  __asm__ volatile("" : "+r"(too_big) : : "memory");
  if (too_big != nullptr || errno != ENOMEM) {
    printf(
        "FAILURE: Expected 8 MiB malloc to fail with ENOMEM (ptr=%p, "
        "errno=%d)\n",
        too_big, errno);
    return 1;
  }

  // Verify Armv8.1-M PMU cycle counter, gettimeofday(), and clock().
  // Advance >= 500,000 cycles (20 ms at 25 MHz = 2 ticks of
  // CLOCKS_PER_SEC=100).
  uint64_t c0 = corstone_get_cycles();
  struct timeval tv0 = {};
  gettimeofday(&tv0, nullptr);
  clock_t t0 = clock();

  uint64_t c1 = c0;
  for (uint32_t iter = 0; iter < 5000000U; ++iter) {
    __asm__ volatile("" : : "r"(iter) : "memory");
    c1 = corstone_get_cycles();
    if (c1 >= c0 + 500000ULL) {
      break;
    }
  }
  struct timeval tv1 = {};
  gettimeofday(&tv1, nullptr);
  clock_t t1 = clock();

  int64_t elapsed_us =
      static_cast<int64_t>(tv1.tv_sec - tv0.tv_sec) * 1000000LL +
      static_cast<int64_t>(tv1.tv_usec - tv0.tv_usec);
  if (c1 < c0 + 500000ULL || elapsed_us < 15000 ||
      t0 == static_cast<clock_t>(-1) || t1 <= t0) {
    printf(
        "FAILURE: Timer did not advance (c0=%llu, c1=%llu, us=%lld, t0=%ld, "
        "t1=%ld)\n",
        static_cast<unsigned long long>(c0),
        static_cast<unsigned long long>(c1), static_cast<long long>(elapsed_us),
        static_cast<long>(t0), static_cast<long>(t1));
    return 1;
  }

  printf("~~~ALL TESTS PASSED~~~\n");
  return 0;
}
