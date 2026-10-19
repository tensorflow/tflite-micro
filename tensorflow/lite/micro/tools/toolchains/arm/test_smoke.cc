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
