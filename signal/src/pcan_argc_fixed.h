#ifndef SIGNAL_MICRO_KERNELS__SRC_PCAN_AGC_FIXED_H_
#define SIGNAL_MICRO_KERNELS__SRC_PCAN_AGC_FIXED_H_
#include <cstdint>

#include "msb.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace tflm_signal {

#define kPcanSnrBits 12
#define kPcanOutputBits 6

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut);

uint32_t PcanShrink(const uint32_t x);

void ApplyPcanAutoGainControlFixed(const int16_t* gain_lut, int32_t snr_shift,
                                   const uint32_t* noise_estimate,
                                   uint32_t* filterbank_output,
                                   int num_channels);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_MICRO_KERNELS__PCAN_AGC_FIXED_H_
