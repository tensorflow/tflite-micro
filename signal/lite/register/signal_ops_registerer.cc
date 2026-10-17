/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/lite/register/signal_ops_registerer.h"

#include "signal/lite/kernels/signal_ops.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace tflm_signal {
namespace lite {

extern "C" void SignalOpsRegisterer(::tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("SignalDelay", Register_DELAY());
  resolver->AddCustom("SignalEnergy", Register_ENERGY());
  resolver->AddCustom("SignalFftAutoScale", Register_FFT_AUTO_SCALE());
  resolver->AddCustom("SignalFilterBank", Register_FILTER_BANK());
  resolver->AddCustom("SignalFilterBankLog", Register_FILTER_BANK_LOG());
  resolver->AddCustom("SignalFilterBankSpectralSubtraction",
                      Register_FILTER_BANK_SPECTRAL_SUBTRACTION());
  resolver->AddCustom("SignalFilterBankSquareRoot",
                      Register_FILTER_BANK_SQUARE_ROOT());
  resolver->AddCustom("SignalFramer", Register_FRAMER());
  resolver->AddCustom("SignalIrfft", Register_IRFFT());
  resolver->AddCustom("SignalOverlapAdd", Register_OVERLAP_ADD());
  resolver->AddCustom("SignalPCAN", Register_PCAN());
  resolver->AddCustom("SignalRfft", Register_RFFT());
  resolver->AddCustom("SignalStacker", Register_STACKER());
  resolver->AddCustom("SignalWindow", Register_WINDOW());
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
