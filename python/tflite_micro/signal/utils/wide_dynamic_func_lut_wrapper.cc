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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace py = pybind11;

constexpr size_t kWideDynamicFunctionBits = 32;
constexpr size_t kWideDynamicFunctionLUTSize =
    (4 * kWideDynamicFunctionBits - 3);

int16_t PcanGainLookupFunction(const float strength, const float offset,
                               const int gain_bits, int32_t input_bits,
                               uint32_t x) {
  const float x_as_float =
      static_cast<float>(x) / (static_cast<uint32_t>(1) << input_bits);
  const float gain_as_float = (static_cast<uint32_t>(1) << gain_bits) *
                              powf(x_as_float + offset, -strength);

  if (gain_as_float > std::numeric_limits<int16_t>::max()) {
    return std::numeric_limits<int16_t>::max();
  }
  return static_cast<int16_t>(gain_as_float + 0.5f);
}

py::list WideDynamicFuncLut(float strength, float offset, int input_bits,
                            int gain_bits) {
  int16_t gain_lut[kWideDynamicFunctionLUTSize + 1];

  gain_lut[0] =
      PcanGainLookupFunction(strength, offset, gain_bits, input_bits, 0);
  gain_lut[1] =
      PcanGainLookupFunction(strength, offset, gain_bits, input_bits, 1);

  for (size_t interval = 2; interval <= kWideDynamicFunctionBits; ++interval) {
    const uint32_t x0 = static_cast<uint32_t>(1) << (interval - 1);
    const uint32_t x1 = x0 + (x0 >> 1);
    const uint32_t x2 =
        (interval == kWideDynamicFunctionBits) ? x0 + (x0 - 1) : 2 * x0;

    const int16_t y0 =
        PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x0);
    const int16_t y1 =
        PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x1);
    const int16_t y2 =
        PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x2);

    const int32_t diff1 = static_cast<int32_t>(y1 - y0);
    const int32_t diff2 = static_cast<int32_t>(y2 - y0);
    const int32_t a1 = 4 * diff1 - diff2;
    const int32_t a2 = diff2 - a1;

    const size_t base_idx = 4 * interval - 6;
    gain_lut[base_idx] = y0;
    gain_lut[base_idx + 1] = static_cast<int16_t>(a1);
    gain_lut[base_idx + 2] = static_cast<int16_t>(a2);
    gain_lut[base_idx + 3] = 0;
  }

  py::list lut_list = py::list();
  for (size_t i = 0; i < kWideDynamicFunctionLUTSize; i++) {
    lut_list.append(gain_lut[i]);
  }

  return lut_list;
}

PYBIND11_MODULE(wide_dynamic_func_lut_wrapper, m) {
  m.doc() = "wide_dynamic_func_lut";
  m.def("wide_dynamic_func_lut", &WideDynamicFuncLut, py::arg("strength"),
        py::arg("offset"), py::arg("input_bits"), py::arg("gain_bits"));
}
