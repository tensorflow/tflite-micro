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
# =============================================================================
"""
Audio feature generation testing, using the AudioPreprocessor class

Run:
bazel build tensorflow/lite/micro/examples/micro_speech:audio_preprocessor_test
bazel-bin/tensorflow/lite/micro/examples/micro_speech/audio_preprocessor_test
"""

from pathlib import Path
import filecmp
import numpy as np

import unittest
import os

import tensorflow as tf
from tflite_micro.tensorflow.lite.micro.examples.micro_speech import audio_preprocessor


class AudioPreprocessorTest(unittest.TestCase):

  def setUp(self):
    self.sample_prefix_path = os.path.join(os.path.dirname(__file__),
                                           'testdata')

  def testFeatureGeneration(self):
    feature_params = audio_preprocessor.FeatureParams()
    audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
    window_size = int(feature_params.window_size_ms *
                      feature_params.sample_rate / 1000)
    data: tf.Tensor = tf.random.uniform(minval=int(tf.dtypes.int16.min),
                                        maxval=tf.dtypes.int16.max,
                                        seed=42,
                                        shape=(1, window_size),
                                        dtype=tf.int32)
    data = tf.cast(data, dtype=tf.int16)  # type: ignore

    # test signal ops internal state retained and features do not match
    feature_eager1 = audio_pp.generate_feature(data)
    feature_eager2 = audio_pp.generate_feature(data)
    self.assertFalse(
        np.array_equal(feature_eager1.numpy(), feature_eager2.numpy()))

    # test eager vs graph execution feature match
    _ = audio_pp.generate_feature_using_graph(data)
    feature_graph = audio_pp.generate_feature_using_graph(data)
    np.testing.assert_array_equal(feature_graph.numpy(),
                                  feature_eager2.numpy())

    # test eager vs MicroInterpreter execution feature match
    feature_tflm = audio_pp.generate_feature_using_tflm(data)
    np.testing.assert_array_equal(feature_tflm.numpy(), feature_eager1.numpy())

    # test signal ops internal state reset
    audio_pp.reset_tflm()
    feature_tflm = audio_pp.generate_feature_using_tflm(data)
    np.testing.assert_array_equal(feature_tflm.numpy(), feature_eager1.numpy())

    # test signal ops internal state retained
    feature_tflm = audio_pp.generate_feature_using_tflm(data)
    np.testing.assert_array_equal(feature_tflm.numpy(), feature_eager2.numpy())

  def testFeatureOutputYes(self):
    feature_params = audio_preprocessor.FeatureParams()
    audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
    audio_pp.load_samples(Path(self.sample_prefix_path, 'yes_30ms.wav'))
    feature = audio_pp.generate_feature_using_tflm(audio_pp.samples)
    feature_list = feature.numpy().tolist()
    expected = [
        124, 105, 126, 103, 125, 101, 123, 100, 116, 98, 115, 97, 113, 90, 91,
        82, 104, 96, 117, 97, 121, 103, 126, 101, 125, 104, 126, 104, 125, 101,
        116, 90, 81, 74, 80, 71, 83, 76, 82, 71
    ]
    self.assertSequenceEqual(feature_list, expected)

  def testFeatureOutputNo(self):
    feature_params = audio_preprocessor.FeatureParams()
    audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
    audio_pp.load_samples(Path(self.sample_prefix_path, 'no_30ms.wav'))
    feature = audio_pp.generate_feature_using_tflm(audio_pp.samples)
    feature_list = feature.numpy().tolist()
    expected = [
        126, 103, 124, 102, 124, 102, 123, 100, 118, 97, 118, 100, 118, 98,
        121, 100, 121, 98, 117, 91, 96, 74, 54, 87, 100, 87, 109, 92, 91, 80,
        64, 55, 83, 74, 74, 78, 114, 95, 101, 81
    ]
    self.assertSequenceEqual(feature_list, expected)


import sys
from absl import flags

if __name__ == '__main__':
  flags.FLAGS(sys.argv)
  unittest.main()
