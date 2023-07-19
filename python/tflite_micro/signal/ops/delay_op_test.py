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
"""Tests for delay op."""

import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro.signal.ops import delay_op
from tflite_micro.python.tflite_micro.signal.utils import util


class DelayOpTest(tf.test.TestCase):

  def TestHelper(self, input_signal, delay_length, frame_size):
    inner_dim_size = input_signal.shape[-1]
    input_signal_rank = len(input_signal.shape)
    frame_num = int(np.ceil((inner_dim_size + delay_length) / frame_size))
    # We need to continue feeding the op with zeros until the delay line is
    # flushed. Pad the input signal to a multiple of frame_size.
    padded_size = frame_num * frame_size
    pad_size = int(padded_size - inner_dim_size)
    # Axes to pass to np.pad. All axes have no padding except the innermost one.
    pad_outer_axes = np.zeros([input_signal_rank - 1, 2], dtype=int)
    pad_input_signal = np.vstack([pad_outer_axes, [0, pad_size]])
    input_signal_padded = np.pad(input_signal, pad_input_signal)
    delay_exp_signal = np.vstack(
        [pad_outer_axes, [delay_length, pad_size - delay_length]])
    delay_exp = np.pad(input_signal, delay_exp_signal)
    delay_out = np.zeros(input_signal_padded.shape)

    in_frame_shape = input_signal.shape[:-1] + (frame_size, )
    func = tf.function(delay_op.delay)
    concrete_function = func.get_concrete_function(tf.TensorSpec(
        in_frame_shape, dtype=tf.int16),
                                                   delay_length=delay_length)
    interpreter = util.get_tflm_interpreter(concrete_function, func)

    for i in range(frame_num):
      in_frame = input_signal_padded[..., i * frame_size:(i + 1) * frame_size]
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_frame_tflm = interpreter.get_output(0)
      # TF
      out_frame = self.evaluate(
          delay_op.delay(in_frame, delay_length=delay_length))
      delay_out[..., i * frame_size:(i + 1) * frame_size] = out_frame
      self.assertAllEqual(out_frame, out_frame_tflm)
    self.assertAllEqual(delay_out, delay_exp)

  def testFrameLargerThanDelay(self):
    self.TestHelper(np.arange(0, 30, dtype=np.int16), 7, 10)

  def testFrameSmallerThanDelay(self):
    self.TestHelper(np.arange(0, 70, dtype=np.int16), 21, 3)

  def testZeroDelay(self):
    self.TestHelper(np.arange(0, 20, dtype=np.int16), 0, 3)

  def testNegativeDelay(self):
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.TestHelper(np.arange(1, 20, dtype=np.int16), -21, 3)

  def testMultiDimensionalDelay(self):
    input_signal = np.reshape(np.arange(0, 120, dtype=np.int16), [2, 3, 20])
    self.TestHelper(input_signal, 4, 6)
    input_signal = np.reshape(np.arange(0, 72, dtype=np.int16),
                              [2, 2, 3, 3, 2])
    self.TestHelper(input_signal, 7, 3)


if __name__ == '__main__':
  tf.test.main()
