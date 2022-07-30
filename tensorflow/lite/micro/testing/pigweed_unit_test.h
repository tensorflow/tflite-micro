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

// Include this header to define unit tests using Pigweed's unit test module,
// pw_unit_test. This header includes all of the necessary headers from Pigweed
// and elsewhere. Including this header and integration into the build system,
// e.g., via the Makefile helper function `pigweed_unit_test,` is all that's
// required to define a unit test. Test definition and assertion macros operate
// roughly like those in Google Test. See the Pigweed documentation and example
// tests elsewhere in tflite-micro.

#ifndef TENSORFLOW_LITE_MICRO_TESTING_PIGWEED_UNIT_TEST_H_
#define TENSORFLOW_LITE_MICRO_TESTING_PIGWEED_UNIT_TEST_H_

#include "pw_unit_test/framework.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Define EXPECT_NEAR in terms of ADD_FAILURE, because Pigweed does not have an
// EXPECT_NEAR. Copy the implementation from micro_test.
#define EXPECT_NEAR(x, y, epsilon)                                            \
  do {                                                                        \
    auto vx = (x);                                                            \
    auto vy = (y);                                                            \
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));               \
    if (vx != vy && delta > epsilon) {                                        \
      MicroPrintf(#x " (%f) near " #y " (%f) failed at %s:%d",                \
                  static_cast<double>(vx), static_cast<double>(vy), __FILE__, \
                  __LINE__);                                                  \
      ADD_FAILURE();                                                          \
    }                                                                         \
  } while (false)

#endif // TENSORFLOW_LITE_MICRO_TESTING_PIGWEED_UNIT_TEST_H_
