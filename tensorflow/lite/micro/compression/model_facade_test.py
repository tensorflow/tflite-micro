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

from tensorflow.lite.python import schema_py_generated as tflite

import model_facade
import test_models

TEST_MODEL = {
    "operator_codes": {
        0: {
            "builtin_code": tflite.BuiltinOperator.FULLY_CONNECTED,
        },
        1: {
            "builtin_code": tflite.BuiltinOperator.ADD,
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
                    "buffer": 1,
                },
                2: {
                    "name": "tensor2",
                    "shape": (4, 1),
                    "type": tflite.TensorType.INT32,
                    "buffer": 1,
                },
                3: {
                    "name": "tensor3",
                    "shape": (2, 1),
                    "type": tflite.TensorType.INT64,
                    "buffer": 1,
                },
            },
        },
    },
    "buffers": {
        0:
        bytes(),
        1:
        bytes((206, 185, 109, 109, 212, 205, 25, 47, 42, 209, 94, 138, 182, 3,
               76, 2)),
    }
}


class TestModelFacade(tf.test.TestCase):

  def setUp(self):
    self.flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(self.flatbuffer)

  def testReadAndWrite(self):
    self.assertEqual(self.flatbuffer, self.facade.pack())

  def testSubgraphIteration(self):
    self.assertEqual(len(self.facade.subgraphs), len(TEST_MODEL["subgraphs"]))
    for i, subgraph in enumerate(self.facade.subgraphs):
      self.assertEqual(i, subgraph.index)


class TestTensors(tf.test.TestCase):

  def setUp(self):
    flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(flatbuffer)

  def testName(self):
    name = TEST_MODEL["subgraphs"][0]["tensors"][0]["name"]
    self.assertEqual(name, self.facade.subgraphs[0].tensors[0].name)

  def testNameIsString(self):
    self.assertTrue(type(self.facade.subgraphs[0].tensors[0].name) is str)

  def testValuesInt8(self):
    tensor_id = 0
    tensor = self.facade.subgraphs[0].tensors[tensor_id]
    self.assertEqual(tensor.type, tflite.TensorType.INT8)

    expect = []
    buffer_id = TEST_MODEL["subgraphs"][0]["tensors"][tensor_id]["buffer"]
    data = TEST_MODEL["buffers"][buffer_id]
    for octet in data:
      expect.append((octet - 0x100) if (octet & 0x80) else octet)

    self.assertAllEqual(tensor.values, expect)

  def testValuesInt16(self):
    tensor_id = 1
    tensor = self.facade.subgraphs[0].tensors[tensor_id]
    self.assertEqual(tensor.type, tflite.TensorType.INT16)

    expect = []
    buffer_id = TEST_MODEL["subgraphs"][0]["tensors"][tensor_id]["buffer"]
    data = TEST_MODEL["buffers"][buffer_id]
    for octet in range(0, len(data), 2):
      value = (data[octet + 1] << 8) + data[octet]
      expect.append((value - 0x1_0000) if (value & 0x8000) else value)

    self.assertAllEqual(tensor.values, expect)

  def testValuesInt32(self):
    tensor_id = 2
    tensor = self.facade.subgraphs[0].tensors[tensor_id]
    self.assertEqual(tensor.type, tflite.TensorType.INT32)

    expect = []
    buffer_id = TEST_MODEL["subgraphs"][0]["tensors"][tensor_id]["buffer"]
    data = TEST_MODEL["buffers"][buffer_id]
    for octet in range(0, len(data), 4):
      value = (data[octet + 3] << 24) + (data[octet + 2] << 16) + (
          data[octet + 1] << 8) + data[octet]
      expect.append((value - 0x1_0000_0000) if (value
                                                & 0x8000_0000) else value)

    self.assertAllEqual(tensor.values, expect)

  def testValuesInt64(self):
    tensor_id = 3
    tensor = self.facade.subgraphs[0].tensors[tensor_id]
    self.assertEqual(tensor.type, tflite.TensorType.INT64)

    expect = []
    buffer_id = TEST_MODEL["subgraphs"][0]["tensors"][tensor_id]["buffer"]
    data = TEST_MODEL["buffers"][buffer_id]
    size = 8
    for octet in range(0, len(data), size):
      value = sum(data[octet + i] << 8 * i for i in range(0, size))
      expect.append((value - 0x1_0000_0000_0000_0000) if (
          value
          & 0x8000_0000_0000_0000) else value)

    self.assertAllEqual(tensor.values, expect)

  def testNormalizeShape(self):
    tensor_id = 3
    tensor = self.facade.subgraphs[0].tensors[tensor_id]
    expect = [1, 1, 2, 1]
    self.assertTrue(len(tensor.shape) < 4)
    self.assertAllEqual(tensor.shape_nhwc, expect)


if __name__ == "__main__":
  tf.test.main()
