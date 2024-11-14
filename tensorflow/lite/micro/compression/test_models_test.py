# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

import test_models
from tensorflow.lite.python import schema_py_generated as tflite


class TestBuild(tf.test.TestCase):

  def setUp(self):
    self.flatbuffer = test_models.build(test_models.EXAMPLE_MODEL)

  def testNotDegenerate(self):
    self.assertTrue(len(self.flatbuffer) > 50)


class TestSpecManipulation(tf.test.TestCase):

  def testGetBuffer(self):
    buffer = test_models.get_buffer(test_models.EXAMPLE_MODEL,
                                    subgraph=0,
                                    tensor=0)
    expect = test_models.EXAMPLE_MODEL["buffers"][1]
    self.assertTrue(buffer is expect)


if __name__ == "__main__":
  tf.test.main()
