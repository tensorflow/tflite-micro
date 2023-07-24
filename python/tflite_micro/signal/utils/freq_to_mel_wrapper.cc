/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace tflite {

// Convert a `freq` in Hz to its value on the Mel scale.
// See: https://en.wikipedia.org/wiki/Mel_scale
// This function is only intended to be used wrapped as the python freq_to_mel
// Why can't we just implement it in Python/numpy?
// The original "Speech Micro" code is written in C and uses 32-bit 'float'
// C types. Python's builtin floating point type is 64-bit wide, which results
// in small differences in the output of the Python and C log() functions.
// A Py wrapper is used in order to establish bit exactness with "Speech Micro",
// while recognizing the slight loss in precision.
float FreqToMel(float freq) { return 1127.0f * log1pf(freq / 700.0f); }

}  // namespace tflite

PYBIND11_MODULE(freq_to_mel_wrapper, m) {
  m.doc() = "freq_to_mel_wrapper";
  m.def("freq_to_mel", &tflite::FreqToMel,
        "Convert a `freq` in Hz to its value on the Mel scale.",
        py::arg("freq"));
}
