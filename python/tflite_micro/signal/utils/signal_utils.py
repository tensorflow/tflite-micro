# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Signal audio utility functions using _runtime.so via ctypes."""

import ctypes
from tflite_micro.python.tflite_micro.runtime import _lib

_lib.TFLM_FreqToMel.restype = ctypes.c_float
_lib.TFLM_FreqToMel.argtypes = [ctypes.c_float]

_lib.TFLM_WideDynamicFuncLut.restype = ctypes.c_int
_lib.TFLM_WideDynamicFuncLut.argtypes = [
  ctypes.c_float,
  ctypes.c_float,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int16),
  ctypes.c_size_t,
  ctypes.POINTER(ctypes.c_size_t),
]


def freq_to_mel(freq: float) -> float:
  """Convert a frequency in Hz to Mel scale."""
  return float(_lib.TFLM_FreqToMel(float(freq)))


def wide_dynamic_func_lut(
  strength: float, offset: float, input_bits: int, gain_bits: int
) -> list[int]:
  """Generate lookup table for PCAN gain function."""
  max_size = 130
  out_lut = (ctypes.c_int16 * max_size)()
  out_size = ctypes.c_size_t(0)
  status = _lib.TFLM_WideDynamicFuncLut(
    float(strength),
    float(offset),
    int(input_bits),
    int(gain_bits),
    out_lut,
    max_size,
    ctypes.byref(out_size),
  )
  if status != 0:
    raise RuntimeError("Failed to compute wide dynamic func LUT")
  return [int(out_lut[i]) for i in range(out_size.value)]
