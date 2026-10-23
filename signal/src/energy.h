/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SIGNAL_ENERGY_H_
#define SIGNAL_ENERGY_H_

#include <stdint.h>

#include "signal/src/complex.h"

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

// Calculates the power spectrum from a DFT output in [start_index, end_index)
// (clamped to [0, output_size]) and zeros elements outside that range.
//
// * `input` and `output` must each have at least `output_size` elements.
void SpectrumToEnergy(const Complex<int16_t>* input, int start_index,
                      int end_index, int output_size, uint32_t* output);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_ENERGY_H_
