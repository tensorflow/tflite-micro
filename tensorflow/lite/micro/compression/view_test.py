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

from absl.testing import absltest
import numpy as np
from tflite_micro.tensorflow.lite.micro.compression import test_models
from tflite_micro.tensorflow.lite.micro.compression import view

_MODEL = {
    "description": "Test model",
    "operator_codes": {
        0: {
            "builtin_code": 0,
        },
        1: {
            "builtin_code": 1,
        },
    },
    "subgraphs": {
        0: {
            "operators": {
                0: {
                    "opcode_index": 1,
                    "inputs": (
                        0,
                        1,
                    ),
                    "outputs": (3, ),
                },
                1: {
                    "opcode_index": 0,
                    "inputs": (
                        3,
                        2,
                    ),
                    "outputs": (4, ),
                },
            },
            "tensors": {
                0: {
                    "shape": (16, 1),
                    "type": 1,
                    "buffer": 1,
                },
                1: {
                    "shape": (16, 1),
                    "type": 1,
                    "buffer": 1,
                },
            },
        },
    },
    "buffers": {
        0: None,
        1: np.array(range(16), dtype="<i1")
    }
}


class UnitTests(absltest.TestCase):

  def testHelloWorld(self):
    self.assertTrue(True)

  def testSmokeTest(self):
    flatbuffer = test_models.build(_MODEL)
    view.create_dictionary(memoryview(flatbuffer))

  def testStrippedDescription(self):
    stripped = _MODEL.copy()
    del stripped["description"]
    flatbuffer = test_models.build(stripped)
    view.create_dictionary(memoryview(flatbuffer))


if __name__ == "__main__":
  absltest.main()
