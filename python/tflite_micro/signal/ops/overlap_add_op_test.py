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
"""Tests for overlap add op."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro.signal.ops import overlap_add_op
from tflite_micro.python.tflite_micro.signal.utils import util


class OverlapAddOpTest(parameterized.TestCase, tf.test.TestCase):

  def RunOverlapAdd(self, interpreter, input_frames, frame_step,
                    expected_output_frames, dtype):
    input_frames = tf.convert_to_tensor(input_frames, dtype=dtype)
    # TFLM
    interpreter.set_input(input_frames, 0)
    interpreter.invoke()
    output_frame = interpreter.get_output(0)
    self.assertAllEqual(output_frame, expected_output_frames)

    # TF
    output_frame = self.evaluate(
        overlap_add_op.overlap_add(input_frames, frame_step))
    self.assertAllEqual(output_frame, expected_output_frames)

  @parameterized.named_parameters(('_FLOAT32InputOutput', tf.float32),
                                  ('_INT16InputOutput', tf.int16))
  def testOverlapAddValidInput(self, dtype):
    input_frames = np.array([[1, -5, 4, 2, 7], [4, 15, -44, 27, -16],
                             [66, -19, 79, 8, -12], [-122, 17, 65, 18, -101],
                             [3, 33, -66, -19, 55]])
    expected_output_frames_step_1 = np.array([[1], [-1], [85], [-183], [133]])
    expected_output_frames_step_2 = np.array([[1, -5], [8, 17], [29, 8],
                                              [-59, 25], [56, 51]])
    expected_output_frames_step_3 = np.array([[1, -5, 4], [6, 22, -44],
                                              [93, -35, 79], [-114, 5, 65],
                                              [21, -68, -66]])
    expected_output_frames_step_4 = np.array([[1, -5, 4, 2], [11, 15, -44, 27],
                                              [50, -19, 79, 8],
                                              [-134, 17, 65, 18],
                                              [-98, 33, -66, -19]])
    expected_output_frames_step_5 = np.array([[1, -5, 4, 2, 7],
                                              [4, 15, -44, 27, -16],
                                              [66, -19, 79, 8, -12],
                                              [-122, 17, 65, 18, -101],
                                              [3, 33, -66, -19, 55]])
    func = tf.function(overlap_add_op.overlap_add)
    # Initialize an interpreter for each step size
    # TODO(b/263020764): use a parameterized test instead
    interpreters = [None] * 6
    for i in range(5):
      interpreters[i] = util.get_tflm_interpreter(
          func.get_concrete_function(
              tf.TensorSpec(np.shape([input_frames[0]]), dtype=dtype), i + 1),
          func)

    frame_num = input_frames.shape[0]
    frame_index = 0
    while frame_index < frame_num:
      self.RunOverlapAdd(interpreters[0], [input_frames[frame_index]],
                         1,
                         expected_output_frames_step_1[frame_index],
                         dtype=dtype)
      self.RunOverlapAdd(interpreters[1], [input_frames[frame_index]],
                         2,
                         expected_output_frames_step_2[frame_index],
                         dtype=dtype)
      self.RunOverlapAdd(interpreters[2], [input_frames[frame_index]],
                         3,
                         expected_output_frames_step_3[frame_index],
                         dtype=dtype)
      self.RunOverlapAdd(interpreters[3], [input_frames[frame_index]],
                         4,
                         expected_output_frames_step_4[frame_index],
                         dtype=dtype)
      self.RunOverlapAdd(interpreters[4], [input_frames[frame_index]],
                         5,
                         expected_output_frames_step_5[frame_index],
                         dtype=dtype)
      frame_index += 1

  @parameterized.named_parameters(('_FLOAT32InputOutput', tf.float32),
                                  ('_INT16InputOutput', tf.int16))
  def testOverlapAddNframes5(self, dtype):
    input_frames = np.array([[1, -5, 4, 2, 7], [4, 15, -44, 27, -16],
                             [66, -19, 79, 8, -12], [-122, 17, 65, 18, -101],
                             [3, 33, -66, -19, 55]])
    expected_output_frames_step_1 = np.array([1, -1, 85, -183, 133])
    expected_output_frames_step_2 = np.array(
        [1, -5, 8, 17, 29, 8, -59, 25, 56, 51])
    expected_output_frames_step_3 = np.array(
        [1, -5, 4, 6, 22, -44, 93, -35, 79, -114, 5, 65, 21, -68, -66])
    expected_output_frames_step_4 = np.array([
        1, -5, 4, 2, 11, 15, -44, 27, 50, -19, 79, 8, -134, 17, 65, 18, -98,
        33, -66, -19
    ])
    expected_output_frames_step_5 = np.array([
        1, -5, 4, 2, 7, 4, 15, -44, 27, -16, 66, -19, 79, 8, -12, -122, 17, 65,
        18, -101, 3, 33, -66, -19, 55
    ])
    func = tf.function(overlap_add_op.overlap_add)
    # Initialize an interpreter for each step size
    # TODO(b/263020764): use a parameterized test instead
    interpreters = [None] * 6
    for i in range(5):
      interpreters[i] = util.get_tflm_interpreter(
          func.get_concrete_function(
              tf.TensorSpec(np.shape(input_frames), dtype=dtype), i + 1), func)
    self.RunOverlapAdd(interpreters[0],
                       input_frames,
                       1,
                       expected_output_frames_step_1,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[1],
                       input_frames,
                       2,
                       expected_output_frames_step_2,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[2],
                       input_frames,
                       3,
                       expected_output_frames_step_3,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[3],
                       input_frames,
                       4,
                       expected_output_frames_step_4,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[4],
                       input_frames,
                       5,
                       expected_output_frames_step_5,
                       dtype=dtype)

  @parameterized.named_parameters(('_FLOAT32InputOutput', tf.float32),
                                  ('_INT16InputOutput', tf.int16))
  def testOverlapAddNframes5Channels2(self, dtype):
    input_frames = np.array([[[1, -5, 4, 2, 7], [4, 15, -44, 27, -16],
                              [66, -19, 79, 8, -12], [-122, 17, 65, 18, -101],
                              [3, 33, -66, -19, 55]],
                             [[1, -5, 4, 2, 7], [4, 15, -44, 27, -16],
                              [66, -19, 79, 8, -12], [-122, 17, 65, 18, -101],
                              [3, 33, -66, -19, 55]]])
    expected_output_frames_step_1 = np.array([[1, -1, 85, -183, 133],
                                              [1, -1, 85, -183, 133]])
    expected_output_frames_step_2 = np.array(
        [[1, -5, 8, 17, 29, 8, -59, 25, 56, 51],
         [1, -5, 8, 17, 29, 8, -59, 25, 56, 51]])
    expected_output_frames_step_3 = np.array(
        [[1, -5, 4, 6, 22, -44, 93, -35, 79, -114, 5, 65, 21, -68, -66],
         [1, -5, 4, 6, 22, -44, 93, -35, 79, -114, 5, 65, 21, -68, -66]])
    expected_output_frames_step_4 = np.array([[
        1, -5, 4, 2, 11, 15, -44, 27, 50, -19, 79, 8, -134, 17, 65, 18, -98,
        33, -66, -19
    ],
                                              [
                                                  1, -5, 4, 2, 11, 15, -44, 27,
                                                  50, -19, 79, 8, -134, 17, 65,
                                                  18, -98, 33, -66, -19
                                              ]])
    expected_output_frames_step_5 = np.array([[
        1, -5, 4, 2, 7, 4, 15, -44, 27, -16, 66, -19, 79, 8, -12, -122, 17, 65,
        18, -101, 3, 33, -66, -19, 55
    ],
                                              [
                                                  1, -5, 4, 2, 7, 4, 15, -44,
                                                  27, -16, 66, -19, 79, 8, -12,
                                                  -122, 17, 65, 18, -101, 3,
                                                  33, -66, -19, 55
                                              ]])
    func = tf.function(overlap_add_op.overlap_add)
    # Initialize an interpreter for each step size
    # TODO(b/263020764): use a parameterized test instead
    interpreters = [None] * 6
    for i in range(5):
      interpreters[i] = util.get_tflm_interpreter(
          func.get_concrete_function(
              tf.TensorSpec(np.shape(input_frames), dtype=dtype), i + 1), func)
    self.RunOverlapAdd(interpreters[0],
                       input_frames,
                       1,
                       expected_output_frames_step_1,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[1],
                       input_frames,
                       2,
                       expected_output_frames_step_2,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[2],
                       input_frames,
                       3,
                       expected_output_frames_step_3,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[3],
                       input_frames,
                       4,
                       expected_output_frames_step_4,
                       dtype=dtype)
    self.RunOverlapAdd(interpreters[4],
                       input_frames,
                       5,
                       expected_output_frames_step_5,
                       dtype=dtype)

  def testStepSizeTooLarge(self):
    ovlerap_add_input = np.zeros(160, dtype=np.int16)
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(overlap_add_op.overlap_add(ovlerap_add_input, 128, 129))

  def testStepSizeNotEqualOutputSize(self):
    ovlerap_add_input = np.zeros(122, dtype=np.int16)
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(overlap_add_op.overlap_add(ovlerap_add_input, 321, 123))


if __name__ == '__main__':
  tf.test.main()
