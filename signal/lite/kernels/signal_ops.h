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

#ifndef SIGNAL_LITE_KERNELS_SIGNAL_OPS_H_
#define SIGNAL_LITE_KERNELS_SIGNAL_OPS_H_

#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {

TfLiteOperator* CreateDelayOperator();
TfLiteOperator* CreateEnergyOperator();
TfLiteOperator* CreateFftAutoScaleOperator();
TfLiteOperator* CreateFilterBankOperator();
TfLiteOperator* CreateFilterBankLogOperator();
TfLiteOperator* CreateFilterBankSpectralSubtractionOperator();
TfLiteOperator* CreateFilterBankSquareRootOperator();
TfLiteOperator* CreateFramerOperator();
TfLiteOperator* CreateIrfftOperator();
TfLiteOperator* CreateOverlapAddOperator();
TfLiteOperator* CreatePcanOperator();
TfLiteOperator* CreateRfftOperator();
TfLiteOperator* CreateStackerOperator();
TfLiteOperator* CreateWindowOperator();

::TfLiteRegistration* Register_DELAY();
::TfLiteRegistration* Register_ENERGY();
::TfLiteRegistration* Register_FFT_AUTO_SCALE();
::TfLiteRegistration* Register_FILTER_BANK();
::TfLiteRegistration* Register_FILTER_BANK_LOG();
::TfLiteRegistration* Register_FILTER_BANK_SPECTRAL_SUBTRACTION();
::TfLiteRegistration* Register_FILTER_BANK_SQUARE_ROOT();
::TfLiteRegistration* Register_FRAMER();
::TfLiteRegistration* Register_IRFFT();
::TfLiteRegistration* Register_OVERLAP_ADD();
::TfLiteRegistration* Register_PCAN();
::TfLiteRegistration* Register_RFFT();
::TfLiteRegistration* Register_STACKER();
::TfLiteRegistration* Register_WINDOW();

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_LITE_KERNELS_SIGNAL_OPS_H_
