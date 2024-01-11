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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_DATA_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
extern const int8_t kConvInput1x32x32x3[];
extern const int8_t kConvFilter8x3x3x3[];
extern const int32_t kConvBiasQuantized8[];
extern const int8_t kConvGoldenOutput1x16x16x8[];

// Kernel Conv Test Cases: Int8Filter1x3x3x1ShouldMatchGolden
extern const int8_t kConvInput1x4x4x1[];
extern const int8_t kConvInput1x5x5x1[];
extern const int8_t kConvFilter1x3x3x1[];
extern const int32_t kConvZeroBias[];
extern const int8_t kConvGoldenOutput4x4InputPaddingSame2x2[];
extern const int8_t kConvGoldenOutput5x5InputPaddingSame3x3[];

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_DATA_H_
