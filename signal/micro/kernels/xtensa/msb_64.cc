/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <xtensa/tie/xt_misc.h>

#include "signal/src/msb.h"

namespace tflite {
namespace tflm_signal {

uint32_t MostSignificantBit64(uint64_t x) {
  // XT_NSAU returns the number of left shifts needed to put the MSB in the
  // leftmost position. Returns 32 if the argument is 0.
  uint32_t upper = 64 - XT_NSAU((uint32_t)(x >> 32));
  if (upper != 32) {
    return upper;
  }
  // Only if the upper bits are all clear do we want to look at the lower bits.
  return 32 - XT_NSAU((uint32_t)x);
}

}  // namespace tflm_signal
}  // namespace tflite
