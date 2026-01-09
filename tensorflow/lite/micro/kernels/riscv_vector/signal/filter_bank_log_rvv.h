#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SIGNAL_FILTER_BANK_LOG_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SIGNAL_FILTER_BANK_LOG_RVV_H_

#include "tensorflow/lite/kernels/internal/common.h"

void FilterbankLogRVV(const uint32_t* input, int num_channels,
                   int32_t output_scale, uint32_t correction_bits,
                   int16_t* output);

#endif // TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SIGNAL_FILTER_BANK_LOG_RVV_H_