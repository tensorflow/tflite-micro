#include "tensorflow/lite/experimental/microfrontend/lib/kiss_fft_common.h"

#define FIXED_POINT 16
namespace kissfft_fixed16 {
// Disable __cplusplus, to avioid extern "C", which disables namepsacing
#undef __cplusplus
#include "kiss_fft.c"
#include "tools/kiss_fftr.c"
#define __cplusplus
}  // namespace kissfft_fixed16
#undef FIXED_POINT
