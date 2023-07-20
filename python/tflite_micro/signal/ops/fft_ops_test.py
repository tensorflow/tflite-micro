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

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import fft_ops
from tflite_micro.python.tflite_micro.signal.utils import util


class RfftOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

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
      self.assertAllEqual(out_frame_exp, out_frame)
      self.assertEqual(scale_exp, scale)
      # TF
      out_frame, scale = self.evaluate(fft_ops.fft_auto_scale(in_frame))
      self.assertAllEqual(out_frame_exp, out_frame)
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
      self.assertAllEqual(out_frame_exp, out_frame)
      # TF
      out_frame = self.evaluate(fft_ops.rfft(in_frame, fft_length))
      self.assertAllEqual(out_frame_exp, out_frame)
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
    self.assertAllEqual(out_frames_exp, out_frame)
    # TF
    out_frames = self.evaluate(fft_ops.rfft(in_frames, fft_length))
    self.assertAllEqual(out_frames_exp, out_frames)

    # Expand outer dims to [4, x, input_size] to test >1 outer dim.
    in_frames_multiple_outer_dims = np.reshape(in_frames, [4, -1, input_size])
    out_frames_exp_multiple_outer_dims = np.reshape(
        out_frames_exp, [4, -1, len(out_frames_exp[0])])
    out_frames_multiple_outer_dims = self.evaluate(
        fft_ops.rfft(in_frames_multiple_outer_dims, fft_length))
    self.assertAllEqual(out_frames_exp_multiple_outer_dims,
                        out_frames_multiple_outer_dims)

  def testRfftOpImpulseTest(self):
    for dtype in [np.int16, np.int32]:
      fft_length = fft_ops._MIN_FFT_LENGTH
      while fft_length <= fft_ops._MAX_FFT_LENGTH:
        max_value = np.iinfo(dtype).max
        # Integer RFFTs are scaled by 1 / fft_length
        expected_real = round(max_value / fft_length)
        expected_imag = 0
        fft_input = np.zeros(fft_length, dtype=dtype)
        fft_input[0] = max_value
        fft_output = self.evaluate(fft_ops.rfft(fft_input, fft_length))
        for i in range(0, int(fft_length / 2 + 1)):
          self.assertEqual(fft_output[2 * i], expected_real)
          self.assertEqual(fft_output[2 * i + 1], expected_imag)
        fft_length = 2 * fft_length

  def testRfftMaxMinAmplitudeTest(self):
    for dtype in [np.int16, np.int32]:
      # Make sure that the FFT doesn't overflow with max/min inputs
      fft_length = fft_ops._MIN_FFT_LENGTH
      while fft_length <= fft_ops._MAX_FFT_LENGTH:
        # Test max
        expected_real = np.iinfo(dtype).max
        expected_imag = 0
        fft_input = expected_real * np.ones(fft_length, dtype=dtype)
        fft_output = self.evaluate(fft_ops.rfft(fft_input, fft_length))
        if dtype == np.int16:
          self.assertAlmostEqual(fft_output[0], expected_real, delta=21)
        elif dtype == np.int32:
          self.assertAlmostEqual(fft_output[0], expected_real, delta=47)
        self.assertAlmostEqual(fft_output[1], expected_imag)
        for i in range(1, int(fft_length / 2 + 1)):
          self.assertEqual(fft_output[2 * i], 0)
          self.assertEqual(fft_output[2 * i + 1], 0)
        # Test min
        expected_real = np.iinfo(dtype).min
        expected_imag = 0
        fft_input = expected_real * np.ones(fft_length, dtype=dtype)
        fft_output = self.evaluate(fft_ops.rfft(fft_input, fft_length))
        self.assertAlmostEqual(fft_output[0], expected_real, delta=22)
        self.assertAlmostEqual(fft_output[1], expected_imag)
        for i in range(1, int(fft_length / 2 + 1)):
          self.assertEqual(fft_output[2 * i], 0)
          self.assertEqual(fft_output[2 * i + 1], 0)
        fft_length = 2 * fft_length

  def testRfftSineTest(self):
    sine_wave_amplitude = 10000
    # how many sine periods per fft_length samples
    sine_wave_angle = (1 / fft_ops._MIN_FFT_LENGTH)
    fft_length = fft_ops._MIN_FFT_LENGTH
    while fft_length <= fft_ops._MAX_FFT_LENGTH:
      fft_input = sine_wave_amplitude * np.sin(
          sine_wave_angle * np.pi * 2 * np.array(range(0, fft_length)))
      fft_input_float = np.float32(fft_input)
      fft_input_int16 = np.int16(np.round(fft_input_float))
      fft_input_int32 = np.int32(np.round(fft_input_float))

      fft_output_float = self.evaluate(
          fft_ops.rfft(fft_input_float, fft_length))
      fft_output_int16 = self.evaluate(
          fft_ops.rfft(fft_input_int16, fft_length))
      fft_output_int32 = np.round(
          self.evaluate(fft_ops.rfft(fft_input_int32, fft_length)))
      sine_bin = round(fft_length / fft_ops._MIN_FFT_LENGTH)
      expected_real = 0
      # The output of floating point RFFT is not scaled
      # This is the expected output of the theorerical DFT
      expected_imag_sine_bin_float = np.float32(-sine_wave_amplitude / 2 *
                                                fft_length)
      # The output of the integer RFFT is scaled by 1 / fft_length
      expected_imag_sine_bin_int16 = np.int16(round(-sine_wave_amplitude / 2))
      expected_imag_sine_bin_int32 = np.int32(round(-sine_wave_amplitude / 2))
      expected_imag_other_bins = 0
      for i in range(0, int(fft_length / 2 + 1)):
        self.assertAlmostEqual(fft_output_float[2 * i],
                               expected_real,
                               delta=0.1)
        self.assertAlmostEqual(fft_output_int16[2 * i], expected_real, delta=2)
        self.assertAlmostEqual(fft_output_int32[2 * i], expected_real, delta=0)
        if i == sine_bin:
          self.assertAlmostEqual(fft_output_float[2 * i + 1],
                                 expected_imag_sine_bin_float,
                                 delta=1.3e-12)
          self.assertAlmostEqual(fft_output_int16[2 * i + 1],
                                 expected_imag_sine_bin_int16,
                                 delta=2)
          self.assertAlmostEqual(fft_output_int32[2 * i + 1],
                                 expected_imag_sine_bin_int32,
                                 delta=2)
        else:
          self.assertAlmostEqual(fft_output_float[2 * i + 1],
                                 expected_imag_other_bins,
                                 delta=0.35)
          self.assertAlmostEqual(fft_output_int16[2 * i + 1],
                                 expected_imag_other_bins,
                                 delta=2)
          self.assertAlmostEqual(fft_output_int32[2 * i + 1],
                                 expected_imag_other_bins,
                                 delta=1)
      fft_length = 2 * fft_length

  def testRfft(self):
    self.SingleRfftTest('testdata/rfft_test1.txt')

  def testRfftLargeOuterDimension(self):
    self.MultiDimRfftTest('testdata/rfft_test1.txt')

  def testFftTooLarge(self):
    for dtype in [np.int16, np.int32, np.float32]:
      fft_input = np.zeros(round(fft_ops._MAX_FFT_LENGTH * 2), dtype=dtype)
      with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
        self.evaluate(
            fft_ops.rfft(fft_input, round(fft_ops._MAX_FFT_LENGTH * 2)))

  def testFftTooSmall(self):
    for dtype in [np.int16, np.int32, np.float32]:
      fft_input = np.zeros(round(fft_ops._MIN_FFT_LENGTH / 2), dtype=dtype)
      with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
        self.evaluate(
            fft_ops.rfft(fft_input, round(fft_ops._MIN_FFT_LENGTH / 2)))

  def testFftLengthNoEven(self):
    for dtype in [np.int16, np.int32, np.float32]:
      fft_input = np.zeros(127, dtype=dtype)
      with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
        self.evaluate(fft_ops.rfft(fft_input, 127))

  def testIrfftTest(self):
    for dtype in [np.int16, np.int32, np.float32]:
      fft_length = fft_ops._MIN_FFT_LENGTH
      while fft_length <= fft_ops._MAX_FFT_LENGTH:
        if dtype == np.float32:
          # Random input in the range [-1, 1)
          fft_input = np.random.random(fft_length).astype(dtype) * 2 - 1
        else:
          fft_input = np.random.randint(
              np.iinfo(np.int16).min,
              np.iinfo(np.int16).max + 1, fft_length).astype(dtype)
        fft_output = self.evaluate(fft_ops.rfft(fft_input, fft_length))
        self.assertEqual(fft_output.shape[0], (fft_length / 2 + 1) * 2)
        ifft_output = self.evaluate(fft_ops.irfft(fft_output, fft_length))
        self.assertEqual(ifft_output.shape[0], fft_length)
        # Output of integer RFFT and IRFFT is scaled by 1/fft_length
        if dtype == np.int16:
          self.assertArrayNear(fft_input,
                               ifft_output.astype(np.int32) * fft_length, 6500)
        elif dtype == np.int32:
          self.assertArrayNear(fft_input,
                               ifft_output.astype(np.int32) * fft_length, 7875)
        else:
          self.assertArrayNear(fft_input, ifft_output, 5e-7)
        fft_length = 2 * fft_length

  def testIrfftLargeOuterDimension(self):
    for dtype in [np.int16, np.int32, np.float32]:
      fft_length = fft_ops._MIN_FFT_LENGTH
      while fft_length <= fft_ops._MAX_FFT_LENGTH:
        if dtype == np.float32:
          # Random input in the range [-1, 1)
          fft_input = np.random.random([2, 5, fft_length
                                        ]).astype(dtype) * 2 - 1
        else:
          fft_input = np.random.randint(
              np.iinfo(np.int16).min,
              np.iinfo(np.int16).max + 1, [2, 5, fft_length]).astype(dtype)
        fft_output = self.evaluate(fft_ops.rfft(fft_input, fft_length))
        self.assertEqual(fft_output.shape[-1], (fft_length / 2 + 1) * 2)
        ifft_output = self.evaluate(fft_ops.irfft(fft_output, fft_length))
        self.assertEqual(ifft_output.shape[-1], fft_length)
        # Output of integer RFFT and IRFFT is scaled by 1/fft_length
        if dtype == np.int16:
          self.assertAllClose(fft_input,
                              ifft_output.astype(np.int32) * fft_length,
                              atol=7875)
        elif dtype == np.int32:
          self.assertAllClose(fft_input,
                              ifft_output.astype(np.int32) * fft_length,
                              atol=7875)
        else:
          self.assertAllClose(fft_input, ifft_output, rtol=5e-7, atol=5e-7)
        fft_length = 2 * fft_length

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
  tf.test.main()
