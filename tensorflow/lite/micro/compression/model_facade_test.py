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

import numpy as np
import tensorflow as tf
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite
from tflite_micro.tensorflow.lite.micro.compression import model_facade
from tflite_micro.tensorflow.lite.micro.compression import test_models

TEST_MODEL = {
    "operator_codes": {
        0: {
            "builtin_code": tflite.BuiltinOperator.FULLY_CONNECTED,
        },
        1: {
            "builtin_code": tflite.BuiltinOperator.ADD,
        },
    },
    "metadata": {
        0: {
            "name": "metadata0",
            "buffer": 0
        },
        1: {
            "name": "metadata1",
            "buffer": 0
        },
    },
    "subgraphs": {
        0: {
            "operators": {
                0: {
                    "opcode_index": 1,  # ADD
                    "inputs": (
                        1,
                        2,
                    ),
                    "outputs": (3, ),
                },
                1: {
                    "opcode_index": 0,  # FULLY_CONNECTED
                    "inputs": (
                        3,
                        4,
                        5,
                    ),
                    "outputs": (6, ),
                },
            },
            "tensors": {
                0: {
                    "name": "tensor0",
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 1,
                },
                1: {
                    "name": "tensor1",
                    "shape": (8, 1),
                    "type": tflite.TensorType.INT16,
                    "buffer": 2,
                },
                2: {
                    "name": "tensor2",
                    "shape": (4, 1),
                    "type": tflite.TensorType.INT32,
                    "buffer": 3,
                },
                3: {
                    "name": "tensor3",
                    "shape": (2, 1),
                    "type": tflite.TensorType.INT64,
                    "buffer": 4,
                },
            },
        },
    },
    "buffers": {
        0: None,
        1: np.array(range(16), dtype=np.dtype("<i1")),
        2: np.array(range(8), dtype=np.dtype("<i2")),
        3: np.array(range(4), dtype=np.dtype("<i4")),
        4: np.array(range(2), dtype=np.dtype("<i8")),
    }
}


class TestModelFacade(tf.test.TestCase):

  def setUp(self):
    self.flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(self.flatbuffer)

  def testLoopback(self):
    self.assertEqual(self.flatbuffer, self.facade.compile())

  def testSubgraphIteration(self):
    self.assertEqual(len(self.facade.subgraphs), len(TEST_MODEL["subgraphs"]))
    for i, subgraph in enumerate(self.facade.subgraphs):
      self.assertEqual(i, subgraph.index)

  def testMetadata(self):
    self.assertIn("metadata0", self.facade.metadata)
    self.assertIn("metadata1", self.facade.metadata)
    self.assertNotIn("metadata2", self.facade.metadata)


class TestTensors(tf.test.TestCase):

  def setUp(self):
    flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(flatbuffer)
    self.test_tensors = TEST_MODEL["subgraphs"][0]["tensors"].items()

  def testName(self):
    for id, attrs in self.test_tensors:
      expect = attrs["name"]
      self.assertEqual(self.facade.subgraphs[0].tensors[id].name, expect)

  def testNameIsString(self):
    for id, _ in self.test_tensors:
      self.assertIsInstance(self.facade.subgraphs[0].tensors[id].name, str)

  def testTensors(self):
    for id, attrs in self.test_tensors:
      tensor = self.facade.subgraphs[0].tensors[id]
      self.assertAllEqual(tensor.shape, attrs["shape"])
      data = TEST_MODEL["buffers"][attrs["buffer"]]
      self.assertAllEqual(tensor.array, data.reshape(tensor.shape))


if __name__ == "__main__":
  tf.test.main()
