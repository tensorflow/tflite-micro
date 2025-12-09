#include <stdint.h>
#include <xtensa/config/core-isa.h>
#include <xtensa/tie/xt_misc.h>

namespace tflite {
namespace tflm_signal {

#if XCHAL_HAVE_HIFI3
#include <xtensa/tie/xt_hifi3.h>
static inline ae_p24x2s MaxAbs16Single(ae_p24x2s max, ae_p24x2s current) {
  return AE_MAXABSSP24S(max, current);
}
#elif XCHAL_HAVE_HIFI_MINI || XCHAL_HAVE_HIFI2 || XCHAL_HAVE_HIFI_EP
#include <xtensa/tie/xt_hifi2.h>
static inline ae_p24x2s MaxAbs16Single(ae_p24x2s max, ae_p24x2s current) {
  current = AE_ABSSP24S(current);
  return AE_MAXP24S(max, current);
}
#endif

#if XCHAL_HAVE_HIFI_MINI || XCHAL_HAVE_HIFI2 || XCHAL_HAVE_HIFI_EP || \
    XCHAL_HAVE_HIFI3
int16_t MaxAbs16(const int16_t* input, int size) {
  int i;
  ae_p24x2s current_24x2 = AE_ZERO24();
  // AE_LP16X2F_IU() effectively pre-increments the address in input_16x2 by 4
  //  bytes before loading, so we need to initialize it accordingly.
  const ae_p16x2s* input_16x2 = (const ae_p16x2s*)(input - 2);
  ae_p24x2s max = AE_ZEROP48();
  const int num_iterations = size / 2;
  for (i = 0; i < num_iterations; i++) {
    // Advancing the pointer by 2 X 16-bits.
    AE_LP16X2F_IU(current_24x2, input_16x2, 4);
    max = MaxAbs16Single(max, current_24x2);
  }
  if (size & 1) {  // n is odd
    // Advancing the pointer by 2 X 16-bits.
    current_24x2 = AE_LP16F_I((ae_p16s*)input_16x2, 4);
    max = MaxAbs16Single(max, current_24x2);
  }
  const int max_L = AE_TRUNCA16P24S_L(max);
  const int max_H = AE_TRUNCA16P24S_H(max);

  return (max_L >= max_H) ? max_L : max_H;
}
#else
// Not a supported Hifi core.
int16_t MaxAbs16(const int16_t* input, int size) {
  int16_t max = 0;
  for (int i = 0; i < size; i++) {
    const int16_t value = input[i];
    if (value > max) {
      max = value;
    } else if (-value > max) {
      max = -value;
    }
  }
  return max;
}
#endif

}  // namespace tflm_signal
}  // namespace tflite
