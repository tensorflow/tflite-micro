#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_RFFT_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_RFFT_RVV_H_

#include "tensorflow/lite/kernels/internal/common.h"

size_t RfftInt16GetNeededMemory(int32_t fft_length);

void* RfftInt16Init(int32_t fft_length, void* state, size_t state_size);

void RfftInt16ApplyRVV(void* state, const int16_t* input,
                    Complex<int16_t>* output);

#endif