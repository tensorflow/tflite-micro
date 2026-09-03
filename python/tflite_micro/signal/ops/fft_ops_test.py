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
"""Tests for FFT ops."""
import os

import numpy as np
import tensorflow as tf

import unittest
from tflite_micro.python.tflite_micro.signal.ops import fft_ops
from tflite_micro.python.tflite_micro.signal.utils import util


class RfftOpTest(unittest.TestCase):

  _PREFIX_PATH = os.path.dirname(__file__)

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def SingleFftAutoScaleTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    func = tf.function(fft_ops.fft_auto_scale)
    input_size = len(lines[0].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.int16))
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    i = 0
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype=np.int16)
      out_frame_exp = [int(j) for j in lines[i + 1].split()]
      scale_exp = [int(j) for j in lines[i + 2].split()]
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      scale = interpreter.get_output(1)
      np.testing.assert_array_equal(out_frame_exp, out_frame)
      self.assertEqual(scale_exp, scale)
      i += 3

  def SingleRfftTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    fft_length = int(args[0])
    func = tf.function(fft_ops.rfft)
    input_size = len(lines[1].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.int16), fft_length)
    # TODO(b/286252893): make test more robust (vs scipy)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in pairs <input, expected>
    i = 1
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype=np.int16)
      out_frame_exp = [int(j) for j in lines[i + 1].split()]
      # Compare TFLM inference against the expected golden values
      # TODO(b/286252893): validate usage of testing vs interpreter here
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      np.testing.assert_array_equal(out_frame_exp, out_frame)
      i += 2

  def MultiDimRfftTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    fft_length = int(args[0])
    func = tf.function(fft_ops.rfft)
    input_size = len(lines[1].split())
    # Since the input starts at line 1, we must add 1. To avoid overflowing,
    # instead subtract 7.
    len_lines_multiple_of_eight = int(len(lines) - len(lines) % 8) - 7
    # Skip line 0, which contains the configuration params.
    # Read lines in pairs <input, expected>
    in_frames = np.array([[int(j) for j in lines[i].split()]
                          for i in range(1, len_lines_multiple_of_eight, 2)],
                         dtype=np.int16)
    out_frames_exp = [[int(j) for j in lines[i + 1].split()]
                      for i in range(1, len_lines_multiple_of_eight, 2)]
    # Compare TFLM inference against the expected golden values
    # TODO(b/286252893): validate usage of testing vs interpreter here
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(np.shape(in_frames), dtype=tf.int16), fft_length)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    interpreter.set_input(in_frames, 0)
    interpreter.invoke()
    out_frame = interpreter.get_output(0)
    np.testing.assert_array_equal(out_frames_exp, out_frame)

  def testRfft(self):
    self.SingleRfftTest('testdata/rfft_test1.txt')

  def testRfftLargeOuterDimension(self):
    self.MultiDimRfftTest('testdata/rfft_test1.txt')

  def testAutoScale(self):
    self.SingleFftAutoScaleTest('testdata/fft_auto_scale_test1.txt')

  def testPow2FftLengthTest(self):
    fft_length, fft_bits = fft_ops.get_pow2_fft_length(131)
    self.assertEqual(fft_length, 256)
    self.assertEqual(fft_bits, 8)
    fft_length, fft_bits = fft_ops.get_pow2_fft_length(73)
    self.assertEqual(fft_length, 128)
    self.assertEqual(fft_bits, 7)
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      fft_ops.get_pow2_fft_length(fft_ops._MIN_FFT_LENGTH / 2 - 1)
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      fft_ops.get_pow2_fft_length(fft_ops._MAX_FFT_LENGTH + 1)


if __name__ == '__main__':
  unittest.main()
