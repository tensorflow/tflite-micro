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

#ifndef SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_COMMON_H_
#define SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_COMMON_H_

// This header file should be included in all variants of kiss_fft_$type.{h,cc}
// so that their sub-included source files do not mistakenly wrap standard C
// header files within their kiss_fft_$type namespaces.
// E.g., This header avoids kiss_fft_int16.h containing:
//   namespace kiss_fft_fixed16 {
//     #include "kiss_fft.h"
//   }
// where kiss_fft.h or kiss_fft.c contains standard C header #includes.
//
// TRICK: By including the following header files here at global scope, their
// preprocessor header guards prevent them from being re-defined inside of the
// kiss_fft_$type namespaces declared within the kiss_fft_$type.{h,cc} sources.
//
// In addition, this header configures unpatched KissFFT for bare-metal TFLM
// builds by overriding KISS_FFT_MALLOC / KISS_FFT_FREE to avoid heap allocation
// references (TFLM always supplies pre-allocated arena buffers) and by
// neutralizing kiss_fft_log.h so KISS_FFT_{ERROR,WARNING,INFO,DEBUG} expand to
// no-ops without pulling in fprintf/stderr or unprefixed log-level macros.
// Note that the original kiss_fft*.h files remain untouched since they may be
// used in libraries that include them directly.

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef KISS_FFT_USE_ALLOCA
#include <alloca.h>
#endif

#ifdef USE_SIMD
#include <xmmintrin.h>
#endif

#ifndef KISS_FFT_MALLOC
#define KISS_FFT_MALLOC(X) (void*)(0x0)
#endif

#ifndef KISS_FFT_FREE
#define KISS_FFT_FREE(X)
#endif

#ifndef kiss_fft_log_h
#define kiss_fft_log_h
#define KISS_FFT_ERROR(...) ((void)0)
#define KISS_FFT_WARNING(...) ((void)0)
#define KISS_FFT_INFO(...) ((void)0)
#define KISS_FFT_DEBUG(...) ((void)0)
#endif

#endif  // SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_COMMON_H_
