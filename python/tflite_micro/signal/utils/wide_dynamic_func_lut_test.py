# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for signal.python.utils.wide_dynamic_func_lut"""
import unittest
from tflite_micro.python.tflite_micro.signal.utils import wide_dynamic_func_lut_wrapper


class WideDynamicFuncLutTest(unittest.TestCase):

  def testWideDynamicFuncLut(self):
    self.maxDiff = None
    expected_lut = [
        32636,
        32633,
        32630,
        -6,
        0,
        0,
        32624,
        -12,
        0,
        0,
        32612,
        -23,
        -2,
        0,
        32587,
        -48,
        0,
        0,
        32539,
        -96,
        0,
        0,
        32443,
        -190,
        0,
        0,
        32253,
        -378,
        4,
        0,
        31879,
        -739,
        18,
        0,
        31158,
        -1409,
        62,
        0,
        29811,
        -2567,
        202,
        0,
        27446,
        -4301,
        562,
        0,
        23707,
        -6265,
        1230,
        0,
        18672,
        -7458,
        1952,
        0,
        13166,
        -7030,
        2212,
        0,
        8348,
        -5342,
        1868,
        0,
        4874,
        -3459,
        1282,
        0,
        2697,
        -2025,
        774,
        0,
        1446,
        -1120,
        436,
        0,
        762,
        -596,
        232,
        0,
        398,
        -313,
        122,
        0,
        207,
        -164,
        64,
        0,
        107,
        -85,
        34,
        0,
        56,
        -45,
        18,
        0,
        29,
        -22,
        8,
        0,
        15,
        -13,
        6,
        0,
        8,
        -8,
        4,
        0,
        4,
        -2,
        0,
        0,
        2,
        -3,
        2,
        0,
        1,
        0,
        0,
        0,
        1,
        -3,
        2,
        0,
        0,
        0,
        0,
    ]
    lut = wide_dynamic_func_lut_wrapper.wide_dynamic_func_lut(
        0.95, 80.0, 7, 21)
    self.assertEqual(lut, expected_lut)


if __name__ == '__main__':
  unittest.main()
