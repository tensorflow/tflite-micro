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
#ifndef SIGNAL_TESTDATA_FFT_TEST_DATA_H_
#define SIGNAL_TESTDATA_FFT_TEST_DATA_H_

#include <cstdint>

namespace tflite {

/* These arrays are generated using random data. They serve to detect changes
 * in the kernels. They do not test correctness.
 */
extern const int16_t kRfftInt16Length512Input[];
extern const int16_t kRfftInt16Length512Golden[];

extern const int32_t kRfftInt32Length512Input[];
extern const int32_t kRfftInt32Length512Golden[];

extern const float kRfftFloatLength512Input[];
extern const float kRfftFloatLength512Golden[];

extern const int16_t kIrfftInt16Length512Input[];
extern const int16_t kIrfftInt16Length512Golden[];

extern const int32_t kIrfftInt32Length512Input[];
extern const int32_t kIrfftInt32Length512Golden[];

extern const float kIrfftFloatLength512Input[];
extern const float kIrfftFloatLength512Golden[];

extern const int16_t kFftAutoScaleLength512Input[];
extern const int16_t kFftAutoScaleLength512Golden[];

}  // namespace tflite

#endif  // SIGNAL_TESTDATA_FFT_TEST_DATA_H_
