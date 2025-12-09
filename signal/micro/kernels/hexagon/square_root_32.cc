/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "signal/src/square_root.h"

extern "C" uint16_t SignalHexagonSqrt32(uint32_t num);

namespace tflite {
namespace tflm_signal {

// SignalHexagonSqrt32() is defined in assembly. This C wrapper is only
// necessary to force TFLM's source specialization to pick up the optimized
// Hexagon implementation instead of the portable one.
uint16_t Sqrt32(uint32_t num) { return SignalHexagonSqrt32(num); }

}  // namespace tflm_signal
}  // namespace tflite
