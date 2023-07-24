# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for energy op."""
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import energy_op
from tflite_micro.python.tflite_micro.signal.utils import util


class EnergyOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def SingleEnergyTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    start_index = int(args[0])
    end_index = int(args[1])

    func = tf.function(energy_op.energy)
    input_size = len(lines[1].split())
    concrete_function = func.get_concrete_function(tf.TensorSpec(
        input_size, dtype=tf.int16),
                                                   start_index=start_index,
                                                   end_index=end_index)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in pairs <input, expected>
    i = 1
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype='int16')
      out_frame_exp = [int(j) for j in lines[i + 1].split()]
      # TFLM
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      for j in range(start_index, end_index):
        self.assertEqual(out_frame_exp[j], out_frame[j])
      # TF
      out_frame = self.evaluate(
          energy_op.energy(in_frame,
                           start_index=start_index,
                           end_index=end_index))
      for j in range(start_index, end_index):
        self.assertEqual(out_frame_exp[j], out_frame[j])
      i += 2

  def testSingleFrame(self):
    start_index = 5
    end_index = 250
    energy_in = [
        -56, 0, 26, 49, 144, -183, -621, 16, 544, 605, 11, -581, -26, 245,
        -210, -273, 200, 541, 268, -319, -43, -544, -747, 356, 415, 356, 174,
        -133, 4, -278, -487, 104, 449, 560, 223, -691, -451, 130, 132, 202, 86,
        -91, 170, -85, -454, -123, 330, 125, -434, 104, 422, 89, -14, -113,
        -123, -63, 125, 142, 40, -218, -183, -10, 3, 154, 95, -64, -108, -55,
        55, 216, 47, -358, -297, 391, 437, 5, -59, -252, -102, -25, -60, 76,
        -46, 6, 128, 113, -4, -101, 20, -75, -154, 88, 144, -50, -163, 58, 112,
        38, 31, 2, -38, -80, 77, 63, -136, -83, 83, 89, 32, 27, 6, -237, -247,
        250, 292, -13, -55, 4, 58, -182, -120, 63, -33, -40, -88, 152, 246, 41,
        -99, -178, -11, 68, -10, 3, 14, 39, 30, -94, -29, 79, -6, -84, -65, 55,
        138, 71, -141, -151, 150, 149, -159, -106, 203, 55, -207, -153, -37,
        231, 187, -6, 54, -66, -85, -258, -244, 271, 157, 24, 117, 144, 144,
        -202, -66, -320, -478, 340, 510, 46, -152, -185, -199, -19, 139, 282,
        -15, -140, 129, 45, -124, -26, 145, -36, -79, -17, -85, -29, 104, 82,
        -84, -7, 127, -96, -210, 60, 114, 67, 40, -3, -1, -101, -76, 77, 55,
        -5, 19, 13, 13, -36, -40, -34, 20, 63, 7, -66, -44, -6, -22, 66, 40,
        -20, 13, 21, -15, -45, 6, 38, 19, -40, -46, -3, 2, 41, 41, -17, -37,
        -11, 15, 13, -4, -5, 0, 1, 2, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1,
        1, -1, -2, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, 1, 0, -1, 0, 1, -1, -1, 0,
        0, 0, -1, 0, -1, 0, 1, 0, 0, -1, -1, 1, -1, -1, 0, 0, 0, 0, -2, -1, -1,
        0, 0, -1, -1, 0, 0, -1, -1, -1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, -1,
        -1, 0, 1, 0, -1, -1, 0, -2, 0, 0, 0, 0, -1, -3, 1, 2, 0, 0, 1, 2, -1,
        -1, -1, -1, -1, -1, 0, 0, 1, 0, -1, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0,
        -1, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 1, -1, -1, 0, -2, 0, 1,
        0, 0, 0, 0, 1, -1, -1, 1, 0, -1, 0, 0, -1, 0, 2, 1, -2, -1, 1, 0, 0,
        -2, 0, 0, -1, -1, 0, 0, 0, 0, -1, -2, -1, 1, 1, 0, 0, 0, 0, -1, -1, 0,
        1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, -1, 0, -1, -1, 0, 0, 0, 1, -1,
        0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, -2, 0, 1, 0, 1, 1, -1, -1, -1,
        0, 0, 0, 0, -1, -1, -1, 0, 1, 0, -2, -1, 1, 1, 1, -1, -3, -1, 1, -1,
        -2, 0, -1, -2, -1, 0, 0, 0, -3, -1, 0, 0, -1, 0, 0, -2, 0
    ]
    energy_exp = [
        0, 0, 0, 0, 0, 337682, 60701, 118629, 332681, 173585, 297785, 684745,
        298961, 47965, 77300, 247985, 515201, 527210, 220301, 58228, 15677,
        36125, 221245, 124525, 199172, 186005, 12965, 19098, 35789, 49124,
        33589, 23725, 13121, 14689, 49681, 130373, 241090, 190994, 66985,
        11029, 9376, 2152, 29153, 10217, 6025, 31460, 23236, 29933, 13988, 965,
        7844, 9898, 25385, 14810, 1753, 56205, 123509, 85433, 3041, 36488,
        18369, 2689, 30848, 62197, 41485, 4745, 109, 1717, 9736, 7082, 7092,
        7250, 24085, 42682, 44701, 36517, 44234, 66258, 54730, 35005, 7272,
        73789, 132977, 25225, 34425, 61540, 106756, 344084, 262216, 57329,
        39962, 98845, 19825, 18666, 16052, 22321, 6530, 8066, 17540, 7105,
        25345, 47700, 17485, 1609, 10202, 11705, 3050, 530, 1465, 2756, 4369,
        4405, 1972, 4840, 2000, 610, 2250, 1480, 1961, 2125, 1685, 1970, 1490,
        394, 41, 1, 4, 0, 0, 1, 0, 0, 1, 2, 5, 1, 0, 1, 1, 1, 1, 1, 2, 0, 1, 1,
        1, 0, 2, 2, 1, 0, 4, 2, 0, 2, 0, 2, 2, 1, 0, 1, 0, 1, 2, 1, 1, 1, 4, 0,
        1, 10, 4, 1, 5, 2, 2, 1, 1, 1, 2, 0, 2, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1,
        2, 4, 1, 0, 0, 2, 2, 1, 0, 1, 5, 5, 1, 4, 0, 2, 0, 0, 5, 2, 1, 0, 1, 1,
        2, 0, 0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 1, 4, 1, 2, 2, 1, 0, 1,
        2, 1, 4, 2, 2, 10, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0
    ]
    energy_out = energy_op.energy(energy_in,
                                  start_index=start_index,
                                  end_index=end_index)

    for j in range(start_index, end_index):
      self.assertEqual(energy_exp[j], energy_out[j])

  def testEnergy(self):
    self.SingleEnergyTest('testdata/energy_test1.txt')


if __name__ == '__main__':
  tf.test.main()
