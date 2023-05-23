#ifndef SIGNAL_MICRO_KERNELS_REGISTER_AUDIO_FRONTEND_OPS_H_
#define SIGNAL_MICRO_KERNELS_REGISTER_AUDIO_FRONTEND_OPS_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
// TODO(b/160234179): Return custom OP registration by value.
TfLiteRegistration_V1 *Register_WINDOW();

}  // namespace tflm_signal
}  // namespace tflite
#endif  // SIGNAL_MICRO_KERNELS_REGISTER_AUDIO_FRONTEND_OPS_H_
