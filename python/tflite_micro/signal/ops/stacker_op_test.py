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
"""Tests for stacker ops."""
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import stacker_op
from tflite_micro.python.tflite_micro.signal.utils import util


class StackerOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def SingleStackerTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    num_channels = int(args[0])
    stacker_left_context = int(args[1])
    stacker_right_context = int(args[2])
    stacker_step = int(args[3])
    func = tf.function(stacker_op.stacker)
    input_size = len(lines[1].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.int16), num_channels,
        stacker_left_context, stacker_right_context, stacker_step)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in triplets <input, expected output, expected valid>
    i = 1
    while i < len(lines):
      input_array = np.array([int(j) for j in lines[i].split()],
                             dtype=np.int16)
      output_array_exp = [int(j) for j in lines[i + 1].split()]
      output_valid_exp = [int(j) for j in lines[i + 2].split()]
      # TFLM
      interpreter.set_input(input_array, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      out_valid = interpreter.get_output(1)
      self.assertEqual(out_valid, output_valid_exp)
      if out_valid:
        self.assertAllEqual(out_frame, output_array_exp)
      # TF
      [out_frame, out_valid] = self.evaluate(
          stacker_op.stacker(input_array, num_channels, stacker_left_context,
                             stacker_right_context, stacker_step))
      self.assertEqual(out_valid, output_valid_exp)
      if out_valid:
        self.assertAllEqual(out_frame, output_array_exp)
      i += 3

  def testStacker(self):
    self.SingleStackerTest('testdata/stacker_test1.txt')


if __name__ == '__main__':
  np.random.seed(0)
  tf.test.main()
