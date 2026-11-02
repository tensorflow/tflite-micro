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

#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"

#undef FIXED_POINT

#define kiss_fft_alloc tflm_kiss_fft_alloc_float
#define kiss_fft tflm_kiss_fft_float
#define kiss_fft_stride tflm_kiss_fft_stride_float
#define kiss_fft_cleanup tflm_kiss_fft_cleanup_float
#define kiss_fft_next_fast_size tflm_kiss_fft_next_fast_size_float
#define kiss_fftr_alloc tflm_kiss_fftr_alloc_float
#define kiss_fftr tflm_kiss_fftr_float
#define kiss_fftri tflm_kiss_fftri_float

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
// Upstream master / Google3 kiss_fft_alloc adds an int-vs-size_t overflow
// check.
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wcast-align"
#endif

namespace kiss_fft_float {
// NOLINTBEGIN(bugprone-suspicious-include)
#include "kiss_fft.c"
#if __has_include("kiss_fftr.c")
#include "kiss_fftr.c"
#else
#include "tools/kiss_fftr.c"
#endif
// NOLINTEND(bugprone-suspicious-include)

#undef kiss_fft_alloc
#undef kiss_fft
#undef kiss_fft_stride
#undef kiss_fft_cleanup
#undef kiss_fft_next_fast_size
#undef kiss_fftr_alloc
#undef kiss_fftr
#undef kiss_fftri

static_assert(sizeof(kiss_fft_cpx) == 2 * sizeof(float),
              "kiss_fft_float must be built with float kiss_fft_scalar");

kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft, void* mem,
                            size_t* lenmem) {
  return tflm_kiss_fft_alloc_float(nfft, inverse_fft, mem, lenmem);
}

void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx* fin, kiss_fft_cpx* fout) {
  tflm_kiss_fft_float(cfg, fin, fout);
}

void kiss_fft_stride(kiss_fft_cfg cfg, const kiss_fft_cpx* fin,
                     kiss_fft_cpx* fout, int fin_stride) {
  tflm_kiss_fft_stride_float(cfg, fin, fout, fin_stride);
}

void kiss_fft_cleanup() { tflm_kiss_fft_cleanup_float(); }

int kiss_fft_next_fast_size(int n) {
  return tflm_kiss_fft_next_fast_size_float(n);
}

kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft, void* mem,
                              size_t* lenmem) {
  return tflm_kiss_fftr_alloc_float(nfft, inverse_fft, mem, lenmem);
}

void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar* timedata,
               kiss_fft_cpx* freqdata) {
  tflm_kiss_fftr_float(cfg, timedata, freqdata);
}

void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx* freqdata,
                kiss_fft_scalar* timedata) {
  tflm_kiss_fftri_float(cfg, freqdata, timedata);
}
}  // namespace kiss_fft_float

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#undef FIXED_POINT
