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

from tensorflow.lite.python import schema_py_generated as tflite

import lib
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
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 2,
                },
                2: {
                    "name": "tensor2",
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 3,
                },
                3: {
                    "name": "tensor3",
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 4,
                },
                4: {
                    "name": "tensor4",
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 5,
                },
                5: {
                    "name": "tensor5",
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 6,
                },
                6: {
                    "name": "tensor6",
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 6,
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
        2:
        bytes((148, 182, 190, 244, 159, 22, 165, 201, 178, 97, 85, 161, 126,
               39, 36, 107)),
        3:
        bytes((67, 84, 53, 155, 137, 191, 63, 251, 102, 53, 123, 189, 34, 212,
               164, 199)),
        4:
        bytes((243, 242, 195, 117, 196, 198, 158, 26, 76, 47, 246, 162, 222,
               94, 6, 255)),
        5:
        bytes((137, 54, 208, 227, 58, 118, 231, 43, 81, 217, 169, 205, 202,
               138, 4, 145)),
        6:
        bytes((234, 181, 174, 210, 0, 49, 101, 145, 0, 13, 167, 230, 86, 78,
               87, 106)),
    }
}


class TestLUT(tf.test.TestCase):

  def setUp(self):
    flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(flatbuffer)

  def testCompressableInputs(self):
    tensors = lib.lut_compressable_inputs(self.facade)
    self.assertEqual(len(tensors), 2)
    names = set(t.name for t in tensors)
    self.assertEqual(names, set(("tensor4", "tensor5")))


class TestBufferValues(tf.test.TestCase):

  def setUp(self):
    flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(flatbuffer)

  def testInt8(self):
    get = lib.buffer_values(self.facade.buffers[1], tflite.TensorType.INT8)
    expect = []
    data = TEST_MODEL["buffers"][1]
    for value in data:
      expect.append((value - 0x100) if (value & 0x80) else value)
    self.assertAllEqual(get, expect)

  def testInt16(self):
    get = lib.buffer_values(self.facade.buffers[1], tflite.TensorType.INT16)
    expect = []
    data = TEST_MODEL["buffers"][1]
    for i in range(0, len(data), 2):
      value = (data[i + 1] << 8) + data[i]
      expect.append((value - 0x1_0000) if (value & 0x8000) else value)
    self.assertAllEqual(get, expect)


class TestTensorValues(tf.test.TestCase):

  def setUp(self):
    flatbuffer = test_models.build(TEST_MODEL)
    self.facade = model_facade.read(flatbuffer)

  def testInt8(self):
    get = lib.tensor_values(self.facade.subgraphs[0].tensors[1])
    expect = []
    data = TEST_MODEL["buffers"][2]
    for value in data:
      expect.append((value - 0x100) if (value & 0x80) else value)
    self.assertAllEqual(get, expect)


if __name__ == "__main__":
  tf.test.main()
