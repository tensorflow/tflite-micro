#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_FILTER_BANK_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_FILTER_BANK_RVV_H_

#include <stdint.h>

#include "tensorflow/lite/kernels/internal/common.h"

struct FilterbankConfig {
  int32_t num_channels;
  const int16_t* channel_frequency_starts;
  const int16_t* channel_weight_starts;
  const int16_t* channel_widths;
  const int16_t* weights;
  const int16_t* unweights;
  int32_t output_scale;

  int32_t input_correction_bits;
};

void FilterbankAccumulateChannelsRVV(const FilterbankConfig* config,
                                  const uint32_t* input, uint64_t* output);

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_FILTER_BANK_RVV_H_