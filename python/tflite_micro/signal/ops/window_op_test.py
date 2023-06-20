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
"""Tests for window ops."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import window_op
from tflite_micro.python.tflite_micro.signal.utils import util


class WindowOpTest(tf.test.TestCase):

  _PREFIX_PATH = resource_loader.get_path_to_datafile('')

  def GetResource(self, filepath):
    full_path = os.path.join(self._PREFIX_PATH, filepath)
    with open(full_path, 'rt') as f:
      file_text = f.read()
    return file_text

  def testWeights(self):
    expected_weights_hann_length_400_shift_12 = [
        0, 1, 2, 3, 5, 8, 11, 14, 18, 23, 28, 33, 39, 46, 53, 60, 68, 77, 86,
        95, 105, 116, 127, 138, 150, 162, 175, 188, 202, 216, 231, 246, 261,
        277, 293, 310, 327, 345, 363, 382, 401, 420, 440, 460, 480, 501, 522,
        544, 566, 589, 611, 634, 658, 682, 706, 730, 755, 780, 806, 831, 857,
        884, 910, 937, 964, 992, 1019, 1047, 1075, 1104, 1133, 1161, 1191,
        1220, 1249, 1279, 1309, 1339, 1369, 1400, 1430, 1461, 1492, 1523, 1554,
        1586, 1617, 1648, 1680, 1712, 1744, 1775, 1807, 1839, 1871, 1903, 1935,
        1968, 2000, 2032, 2064, 2096, 2128, 2161, 2193, 2225, 2257, 2289, 2321,
        2352, 2384, 2416, 2448, 2479, 2510, 2542, 2573, 2604, 2635, 2666, 2696,
        2727, 2757, 2787, 2817, 2847, 2876, 2905, 2935, 2963, 2992, 3021, 3049,
        3077, 3104, 3132, 3159, 3186, 3212, 3239, 3265, 3290, 3316, 3341, 3366,
        3390, 3414, 3438, 3462, 3485, 3507, 3530, 3552, 3574, 3595, 3616, 3636,
        3656, 3676, 3695, 3714, 3733, 3751, 3769, 3786, 3803, 3819, 3835, 3850,
        3865, 3880, 3894, 3908, 3921, 3934, 3946, 3958, 3969, 3980, 3991, 4001,
        4010, 4019, 4028, 4036, 4043, 4050, 4057, 4063, 4068, 4073, 4078, 4082,
        4085, 4088, 4091, 4093, 4094, 4095, 4096, 4096, 4095, 4094, 4093, 4091,
        4088, 4085, 4082, 4078, 4073, 4068, 4063, 4057, 4050, 4043, 4036, 4028,
        4019, 4010, 4001, 3991, 3980, 3969, 3958, 3946, 3934, 3921, 3908, 3894,
        3880, 3865, 3850, 3835, 3819, 3803, 3786, 3769, 3751, 3733, 3714, 3695,
        3676, 3656, 3636, 3616, 3595, 3574, 3552, 3530, 3507, 3485, 3462, 3438,
        3414, 3390, 3366, 3341, 3316, 3290, 3265, 3239, 3212, 3186, 3159, 3132,
        3104, 3077, 3049, 3021, 2992, 2963, 2935, 2905, 2876, 2847, 2817, 2787,
        2757, 2727, 2696, 2666, 2635, 2604, 2573, 2542, 2510, 2479, 2448, 2416,
        2384, 2352, 2321, 2289, 2257, 2225, 2193, 2161, 2128, 2096, 2064, 2032,
        2000, 1968, 1935, 1903, 1871, 1839, 1807, 1775, 1744, 1712, 1680, 1648,
        1617, 1586, 1554, 1523, 1492, 1461, 1430, 1400, 1369, 1339, 1309, 1279,
        1249, 1220, 1191, 1161, 1133, 1104, 1075, 1047, 1019, 992, 964, 937,
        910, 884, 857, 831, 806, 780, 755, 730, 706, 682, 658, 634, 611, 589,
        566, 544, 522, 501, 480, 460, 440, 420, 401, 382, 363, 345, 327, 310,
        293, 277, 261, 246, 231, 216, 202, 188, 175, 162, 150, 138, 127, 116,
        105, 95, 86, 77, 68, 60, 53, 46, 39, 33, 28, 23, 18, 14, 11, 8, 5, 3,
        2, 1, 0
    ]
    weights = window_op.hann_window_weights(400, 12)
    self.assertAllEqual(weights, expected_weights_hann_length_400_shift_12)
    expected_weights_squart_root_hann_cwola_length_256_shift_12 = [
        25, 75, 126, 176, 226, 276, 326, 376, 426, 476, 526, 576, 626, 675,
        725, 774, 824, 873, 922, 971, 1020, 1068, 1117, 1165, 1213, 1261, 1309,
        1356, 1404, 1451, 1498, 1544, 1591, 1637, 1683, 1729, 1774, 1819, 1864,
        1909, 1953, 1997, 2041, 2084, 2127, 2170, 2213, 2255, 2296, 2338, 2379,
        2420, 2460, 2500, 2540, 2579, 2618, 2656, 2694, 2732, 2769, 2806, 2843,
        2878, 2914, 2949, 2984, 3018, 3052, 3085, 3118, 3150, 3182, 3214, 3244,
        3275, 3305, 3334, 3363, 3392, 3420, 3447, 3474, 3500, 3526, 3551, 3576,
        3600, 3624, 3647, 3670, 3692, 3713, 3734, 3755, 3775, 3794, 3812, 3831,
        3848, 3865, 3881, 3897, 3912, 3927, 3941, 3954, 3967, 3979, 3991, 4002,
        4012, 4022, 4031, 4040, 4048, 4055, 4062, 4068, 4074, 4079, 4083, 4087,
        4090, 4092, 4094, 4095, 4096, 4096, 4095, 4094, 4092, 4090, 4087, 4083,
        4079, 4074, 4068, 4062, 4055, 4048, 4040, 4031, 4022, 4012, 4002, 3991,
        3979, 3967, 3954, 3941, 3927, 3912, 3897, 3881, 3865, 3848, 3831, 3812,
        3794, 3775, 3755, 3734, 3713, 3692, 3670, 3647, 3624, 3600, 3576, 3551,
        3526, 3500, 3474, 3447, 3420, 3392, 3363, 3334, 3305, 3275, 3244, 3214,
        3182, 3150, 3118, 3085, 3052, 3018, 2984, 2949, 2914, 2878, 2843, 2806,
        2769, 2732, 2694, 2656, 2618, 2579, 2540, 2500, 2460, 2420, 2379, 2338,
        2296, 2255, 2213, 2170, 2127, 2084, 2041, 1997, 1953, 1909, 1864, 1819,
        1774, 1729, 1683, 1637, 1591, 1544, 1498, 1451, 1404, 1356, 1309, 1261,
        1213, 1165, 1117, 1068, 1020, 971, 922, 873, 824, 774, 725, 675, 626,
        576, 526, 476, 426, 376, 326, 276, 226, 176, 126, 75, 25
    ]
    weights = window_op.square_root_hann_cwola_window_weights(256, 128, 12)
    self.assertAllEqual(
        weights, expected_weights_squart_root_hann_cwola_length_256_shift_12)

  def SingleWindowTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    window_type = args[0]
    dtype = args[1]
    shift = int(args[2])
    func = tf.function(window_op.window)
    input_size = len(lines[1].split())
    self.assertEqual(dtype, 'int16')
    self.assertEqual(window_type, 'hann')
    weights = window_op.hann_window_weights(input_size, shift)
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(input_size, dtype=tf.int16),
        tf.TensorSpec(input_size, dtype=tf.int16),
        shift=shift)
    # TODO(b/286252893): make test more robust (vs scipy)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    # Skip line 0, which contains the configuration params.
    # Read lines in pairs <input, expected>
    i = 1
    while i < len(lines):
      in_frame = np.array([int(j) for j in lines[i].split()], dtype='int16')
      out_frame_exp = [int(j) for j in lines[i + 1].split()]
      # TFLite
      interpreter.set_input(in_frame, 0)
      interpreter.set_input(weights, 1)
      interpreter.invoke()
      out_frame = interpreter.get_output(0)
      self.assertAllEqual(out_frame_exp, out_frame)
      # TF
      out_frame = self.evaluate(
          window_op.window(in_frame, weights, shift=shift))
      self.assertAllEqual(out_frame_exp, out_frame)
      i += 2

  def RunMultiDimWindow(self, shift, dtype, in_frames, weights,
                        out_frames_exp):
    func = tf.function(window_op.window)
    # TFLite
    concrete_function = func.get_concrete_function(
        tf.TensorSpec(np.shape(in_frames), dtype=dtype),
        tf.TensorSpec(np.shape(weights), dtype=dtype),
        shift=shift)
    interpreter = util.get_tflm_interpreter(concrete_function, func)
    interpreter.set_input(in_frames, 0)
    interpreter.set_input(weights, 1)
    interpreter.invoke()
    out_frame = interpreter.get_output(0)
    self.assertAllEqual(out_frames_exp, out_frame)
    # TF
    out_frame = self.evaluate(window_op.window(in_frames, weights,
                                               shift=shift))
    self.assertAllEqual(out_frames_exp, out_frame)

  def MultiDimWindowTest(self, filename):
    lines = self.GetResource(filename).splitlines()
    args = lines[0].split()
    window_type = args[0]
    dtype = args[1]
    shift = int(args[2])
    input_size = len(lines[1].split())
    self.assertEqual(dtype, 'int16')
    self.assertEqual(window_type, 'hann')
    weights = window_op.hann_window_weights(input_size, shift)

    # Since the input starts at line 1, we must add 1. To avoid overflowing,
    # instead subtract 7.
    num_lines_multiple_of_eight = int(len(lines) - len(lines) % 8) - 7
    # Skip line 0, which contains the configuration params.
    # Read lines in pairs <input, expected>
    in_frames = np.array([[int(j) for j in lines[i].split()]
                          for i in range(1, num_lines_multiple_of_eight, 2)],
                         dtype='int16')
    out_frames_exp = [[int(j) for j in lines[i + 1].split()]
                      for i in range(1, num_lines_multiple_of_eight, 2)]
    self.RunMultiDimWindow(shift, dtype, in_frames, weights, out_frames_exp)

    # Expand outer dims to [4, x, input_size] to test >1 outer dim.
    in_frames_multiple_outer_dims = np.reshape(in_frames, [4, -1, input_size])
    out_frames_exp_multiple_outer_dims = np.reshape(out_frames_exp,
                                                    [4, -1, input_size])
    self.RunMultiDimWindow(shift, dtype, in_frames_multiple_outer_dims,
                           weights, out_frames_exp_multiple_outer_dims)

  def testSingleFrame(self):
    frame_in = [
        165, 296, 414, 400, 251, 87, 25, 84, 197, 256, 188, 15, -135, -149,
        -68, -25, -85, -145, -87, 51, 116, 41, -80, -118, -64, 1, 21, 12, 7, 3,
        -6, -9, -9, -44, -129, -210, -192, -60, 72, 70, -65, -178, -151, -55,
        -66, -234, -409, -418, -285, -188, -231, -323, -303, -143, 41, 126, 99,
        29, -6, 27, 109, 165, 130, 28, -41, -20, 28, 10, -68, -92, 6, 153, 222,
        168, 61, -16, -27, 16, 86, 137, 132, 82, 46, 66, 100, 73, -26, -124,
        -153, -121, -82, -72, -92, -120, -120, -53, 79, 206, 221, 83, -113,
        -222, -169, -15, 99, 88, -1, -45, 26, 141, 166, 61, -62, -62, 78, 222,
        233, 119, 5, 14, 159, 346, 463, 454, 349, 233, 183, 211, 247, 204, 68,
        -68, -90, 3, 78, 12, -161, -276, -237, -134, -122, -217, -296, -258,
        -159, -113, -146, -184, -160, -89, -23, 6, -5, -31, -34, 6, 56, 55, 0,
        -33, 23, 135, 198, 157, 47, -46, -75, -44, -8, -19, -102, -203, -239,
        -190, -122, -104, -116, -80, 31, 147, 175, 94, -33, -117, -122, -81,
        -59, -86, -125, -121, -68, -17, -8, -37, -69, -89, -90, -75, -62, -83,
        -141, -181, -151, -80, -48, -80, -113, -78, 3, 51, 27, -29, -54, -30,
        20, 61, 67, 48, 48, 104, 193, 241, 204, 113, 52, 56, 91, 93, 39, -43,
        -115, -145, -117, -38, 61, 122, 100, 12, -63, -55, 39, 143, 186, 163,
        128, 125, 146, 163, 163, 163, 171, 173, 149, 107, 74, 62, 42, -11, -74,
        -79, 0, 100, 126, 63, -9, -13, 44, 107, 131, 111, 47, -48, -138, -175,
        -160, -145, -162, -173, -131, -42, 15, -21, -128, -217, -207, -107,
        -10, 5, -36, -22, 89, 188, 122, -98, -267, -214, -15, 97, -5, -193,
        -251, -122, 52, 107, 33, -49, -25, 97, 219, 259, 213, 136, 88, 81, 90,
        78, 35, -19, -64, -100, -138, -157, -126, -52, -4, -26, -85, -102, -58,
        -6, 12, 12, 23, 22, -28, -101, -108, -23, 72, 80, -9, -116, -169, -164,
        -143, -128, -113, -82, -33, 11, 31, 33, 40, 70, 106, 117, 103, 82, 65,
        44, 13, -9, -7, 10, 27, 47, 83, 122, 138, 122, 93, 59, 11, -41, -42,
        46, 177, 236, 163, 18, -73, -58, -3, -2, -71, -144, -158, -115, -57,
        -23, -10, 0, 22, 48, 57, 35, -2, -31, -22, 29
    ]

    exp_out = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -2, -2, -1, -1, -2, -3, -2, 1, 2,
        1, -3, -4, -3, 0, 0, 0, 0, 0, -1, -1, -1, -3, -10, -16, -16, -6, 6, 6,
        -7, -19, -17, -7, -8, -29, -53, -56, -40, -28, -35, -50, -49, -24, 7,
        22, 18, 5, -2, 5, 22, 35, 28, 6, -10, -5, 6, 2, -18, -25, 1, 43, 64,
        50, 18, -5, -9, 5, 28, 46, 46, 29, 16, 24, 37, 28, -11, -50, -63, -51,
        -35, -32, -41, -54, -55, -25, 37, 98, 107, 41, -57, -114, -88, -8, 53,
        47, -1, -26, 14, 80, 96, 35, -38, -38, 47, 137, 146, 75, 3, 9, 104,
        230, 311, 308, 240, 161, 128, 149, 176, 147, 49, -51, -67, 2, 59, 9,
        -125, -215, -186, -106, -98, -175, -240, -211, -131, -94, -122, -155,
        -136, -76, -20, 5, -5, -28, -30, 5, 49, 49, 0, -30, 20, 123, 181, 144,
        43, -43, -70, -42, -8, -18, -97, -193, -229, -182, -118, -101, -113,
        -78, 30, 143, 170, 92, -33, -116, -121, -80, -59, -86, -124, -121, -68,
        -17, -8, -37, -69, -89, -90, -75, -62, -83, -141, -181, -151, -80, -48,
        -80, -113, -78, 2, 50, 26, -29, -54, -30, 19, 60, 65, 47, 46, 101, 188,
        234, 197, 109, 50, 53, 87, 88, 37, -41, -109, -137, -110, -36, 56, 112,
        92, 10, -58, -50, 35, 128, 166, 144, 113, 109, 127, 141, 140, 139, 145,
        146, 125, 89, 61, 50, 34, -9, -60, -63, 0, 78, 98, 48, -7, -10, 33, 79,
        96, 81, 33, -35, -98, -123, -112, -100, -111, -117, -88, -28, 9, -14,
        -82, -137, -129, -66, -7, 2, -22, -13, 51, 106, 68, -55, -146, -115,
        -8, 50, -3, -98, -125, -60, 24, 50, 15, -23, -12, 42, 94, 110, 89, 55,
        35, 31, 34, 29, 13, -7, -23, -35, -48, -53, -42, -17, -2, -8, -26, -30,
        -17, -2, 3, 3, 5, 5, -7, -24, -25, -6, 15, 16, -2, -23, -33, -31, -26,
        -23, -19, -14, -6, 1, 4, 4, 5, 8, 12, 13, 11, 8, 6, 4, 1, -1, -1, 0, 2,
        3, 5, 7, 8, 6, 4, 2, 0, -2, -2, 1, 5, 7, 4, 0, -2, -2, -1, -1, -2, -2,
        -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0
    ]
    weights = window_op.hann_window_weights(len(frame_in), 12)
    frame_out = window_op.window(frame_in, weights, shift=12)
    self.assertAllEqual(exp_out, frame_out)

  def testWindow(self):
    self.SingleWindowTest('testdata/window_test1.txt')

  def testWindowLargeOuterDimension(self):
    self.MultiDimWindowTest('testdata/window_test1.txt')


if __name__ == '__main__':
  tf.test.main()
