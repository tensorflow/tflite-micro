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

#ifndef SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_INT16_H_
#define SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_INT16_H_

#include "signal/src/kiss_fft_wrappers/kiss_fft_common.h"

// Wrap 16-bit fixed-point KissFFT in its own namespace and rename the
// underlying C symbols so multiple KissFFT precisions (16/32-bit integer,
// float) can coexist in a single translation unit and binary without C-linkage
// collisions.
#undef KISS_FFT_H
#undef KISS_FTR_H
#undef KISS_SUFFIX
#undef KISS_FFT_SHARED
#undef KISS_FFT_API
#undef kiss_fft_scalar
#undef FIXED_POINT
#define FIXED_POINT 16

#define kiss_fft_alloc tflm_kiss_fft_alloc_fixed16
#define kiss_fft tflm_kiss_fft_fixed16
#define kiss_fft_stride tflm_kiss_fft_stride_fixed16
#define kiss_fft_cleanup tflm_kiss_fft_cleanup_fixed16
#define kiss_fft_next_fast_size tflm_kiss_fft_next_fast_size_fixed16
#define kiss_fftr_alloc tflm_kiss_fftr_alloc_fixed16
#define kiss_fftr tflm_kiss_fftr_fixed16
#define kiss_fftri tflm_kiss_fftri_fixed16

namespace kiss_fft_fixed16 {
#include "kiss_fft.h"
#if __has_include("kiss_fftr.h")
#include "kiss_fftr.h"
#else
#include "tools/kiss_fftr.h"
#endif

#undef kiss_fft_alloc
#undef kiss_fft
#undef kiss_fft_stride
#undef kiss_fft_cleanup
#undef kiss_fft_next_fast_size
#undef kiss_fftr_alloc
#undef kiss_fftr
#undef kiss_fftri

static_assert(sizeof(kiss_fft_cpx) == 2 * sizeof(int16_t),
              "kiss_fft_fixed16 must be built with FIXED_POINT=16");

kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft, void* mem,
                            size_t* lenmem);
void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx* fin, kiss_fft_cpx* fout);
void kiss_fft_stride(kiss_fft_cfg cfg, const kiss_fft_cpx* fin,
                     kiss_fft_cpx* fout, int fin_stride);
void kiss_fft_cleanup();
int kiss_fft_next_fast_size(int n);
kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft, void* mem,
                              size_t* lenmem);
void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar* timedata,
               kiss_fft_cpx* freqdata);
void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx* freqdata,
                kiss_fft_scalar* timedata);
}  // namespace kiss_fft_fixed16

#undef FIXED_POINT
#undef KISS_SUFFIX

#endif  // SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_INT16_H_
