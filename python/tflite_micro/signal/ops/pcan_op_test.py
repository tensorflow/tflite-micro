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

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import pcan_op
from tflite_micro.python.tflite_micro.signal.utils import util


class PcanOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def SinglePcanOpTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    strength = float(args[0])
    offset = float(args[1])
    gain_bits = int(args[2])
    smoothing_bits = int(args[3])
    input_correction_bits = int(args[4])

    func = tf.function(pcan_op.pcan)
    channel_num = len(lines[1].split())

    concrete_function = func.get_concrete_function(
        tf.TensorSpec(channel_num, dtype=tf.uint32),
        tf.TensorSpec(channel_num, dtype=tf.uint32),
        strength=strength,
        offset=offset,
        gain_bits=gain_bits,
        smoothing_bits=smoothing_bits,
        input_correction_bits=input_correction_bits)
    interpreter = util.get_tflm_interpreter(concrete_function, func)

    # Read lines in pairs <input, noise_estimate, expected>
    for i in range(1, len(lines), 3):
      in_frame = np.array([int(j) for j in lines[i + 0].split()],
                          dtype='uint32')
      noise_estimate = np.array([int(j) for j in lines[i + 1].split()],
                                dtype='uint32')
      output_expected = np.array([int(j) for j in lines[i + 2].split()],
                                 dtype='uint32')
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.set_input(noise_estimate, 1)
      interpreter.invoke()
      output = interpreter.get_output(0)
      self.assertAllEqual(output_expected, output)
      # TF
      output = self.evaluate(
          pcan_op.pcan(in_frame, noise_estimate, strength, offset, gain_bits,
                       smoothing_bits, input_correction_bits))
      self.assertAllEqual(output_expected, output)

  def testPcanOp(self):
    self.SinglePcanOpTest('testdata/pcan_op_test1.txt')


if __name__ == '__main__':
  tf.test.main()
