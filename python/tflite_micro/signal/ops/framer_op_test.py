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
"""Tests for framer op."""
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import framer_op
from tflite_micro.python.tflite_micro.signal.utils import util


class FramerOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def SingleFramerTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    frame_size = int(args[0])
    frame_step = int(args[1])
    prefill = bool(int(args[2]))
    func = tf.function(framer_op.framer)
    input_size = len(lines[1].split())
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.int16), frame_size, frame_step,
        prefill)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in triplets <input, expected output, expected valid>
    i = 1
    while i < len(lines):
      in_block = np.array([int(j) for j in lines[i].split()], dtype=np.int16)
      out_frame_exp = [[int(j) for j in lines[i + 1].split()]]
      out_valid_exp = [int(j) for j in lines[i + 2].split()]
      # TFLM
      interpreter.set_input(in_block, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      out_valid = interpreter.get_output(1)
      self.assertEqual(out_valid, out_valid_exp)
      if out_valid:
        self.assertAllEqual(out_frame, out_frame_exp)
      # TF
      out_frame, out_valid = self.evaluate(
          framer_op.framer(in_block, frame_size, frame_step, prefill))
      self.assertEqual(out_valid, out_valid_exp)
      if out_valid:
        self.assertAllEqual(out_frame, out_frame_exp)
      i += 3

  def MultiFrameRandomInputFramerTest(self, n_frames):
    # Terminonlogy: input is in blocks, output is in frames
    frame_step = 160
    frame_size = 400
    prefill = True
    block_num = 10
    block_size = frame_step * n_frames

    test_input = np.random.randint(np.iinfo('int16').min,
                                   np.iinfo('int16').max,
                                   block_size * block_num,
                                   dtype=np.int16)
    expected_output = np.concatenate((np.zeros(frame_size - frame_step,
                                               dtype=np.int16), test_input))
    func = tf.function(framer_op.framer)
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(block_size, dtype=tf.int16), frame_size, frame_step,
        prefill)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    block_index = 0
    frame_index = 0
    while block_index < block_num:
      in_block = test_input[(block_index * block_size):((block_index + 1) *
                                                        block_size)]
      expected_valid = 1
      expected_frame = [
          expected_output[((frame_index + i) *
                           frame_step):((frame_index + i) * frame_step +
                                        frame_size)] for i in range(n_frames)
      ]
      # TFLM
      interpreter.set_input(in_block, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      out_valid = interpreter.get_output(1)
      self.assertEqual(out_valid, expected_valid)
      if out_valid:
        self.assertAllEqual(out_frame, expected_frame)
      # TF
      out_frame, out_valid = self.evaluate(
          framer_op.framer(in_block, frame_size, frame_step, prefill))
      frame_index += n_frames
      self.assertEqual(out_valid, expected_valid)
      self.assertAllEqual(out_frame, expected_frame)
      block_index += 1

  def testFramerVectors(self):
    self.SingleFramerTest('testdata/framer_test1.txt')

  def testFramerRandomInput(self):
    self.MultiFrameRandomInputFramerTest(1)

  def testFramerRandomInputNframes2(self):
    self.MultiFrameRandomInputFramerTest(2)

  def testFramerRandomInputNframes4(self):
    self.MultiFrameRandomInputFramerTest(4)

  def testStepSizeTooLarge(self):
    framer_input = np.zeros(160, dtype=np.int16)
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(framer_op.framer(framer_input, 128, 129))

  def testStepSizeNotEqualInputSize(self):
    framer_input = np.zeros(122, dtype=np.int16)
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(framer_op.framer(framer_input, 321, 123))


if __name__ == '__main__':
  np.random.seed(0)
  tf.test.main()
