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
#include <stdio.h>

#if defined(__ARM_FEATURE_MVE)
#include <arm_mve.h>
#endif

int main(int argc, char** argv) {
  printf("Hello from Cortex-M bare-metal under QEMU!\n");

#if defined(__ARM_FEATURE_MVE)
  // Helium / MVE Vector addition
  int32x4_t a = vdupq_n_s32(10);
  int32x4_t b = vdupq_n_s32(32);
  int32x4_t c = vaddq_s32(a, b);

  int lane0 = vgetq_lane_s32(c, 0);
  printf("Helium MVE 10 + 32 = %d\n", lane0);
#else
  int lane0 = 10 + 32;
  printf("Scalar ALU 10 + 32 = %d\n", lane0);
#endif

  if (lane0 == 42) {
    printf("SUCCESS: Vector/Scalar addition passed!\n");
    return 0;
  } else {
    printf("FAILURE: Unexpected value %d\n", lane0);
    return 1;
  }
}
