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
#
"""Utilities for building test models"""

import flatbuffers

from tensorflow.lite.python import schema_py_generated as tflite


def build(spec: dict) -> bytearray:
  """Build a tflite flatbuffer from a model spec.

  Args:
    spec: A dictionary representation of the model, a prototype of which
      can be found in the EXAMPLE_MODEL attribute of this module.

  Returns:
    A tflite flatbuffer.
  """
  root = tflite.ModelT()

  root.operatorCodes = []
  for id, operator_code in spec["operator_codes"].items():
    assert id == len(root.operatorCodes)
    opcode_t = tflite.OperatorCodeT()
    root.operatorCodes.append(opcode_t)
    opcode_t.builtinCode = operator_code["builtin_code"]

  root.subgraphs = []
  for id, subgraph in spec["subgraphs"].items():
    assert id == len(root.subgraphs)
    subgraph_t = tflite.SubGraphT()
    root.subgraphs.append(subgraph_t)

    subgraph_t.operators = []
    for id, operator in subgraph["operators"].items():
      assert id == len(subgraph_t.operators)
      operator_t = tflite.OperatorT()
      operator_t.opcodeIndex = operator["opcode_index"]
      operator_t.inputs = operator["inputs"]
      operator_t.outputs = operator["outputs"]
      subgraph_t.operators.append(operator_t)

    subgraph_t.tensors = []
    for id, tensor in subgraph["tensors"].items():
      assert id == len(subgraph_t.tensors)
      tensor_t = tflite.TensorT()
      tensor_t.name = tensor.get("name", f"tensor{id}")
      tensor_t.shape = tensor["shape"]
      tensor_t.type = tensor["type"]
      tensor_t.buffer = tensor["buffer"]
      subgraph_t.tensors.append(tensor_t)

  root.buffers = []
  for id, data in spec["buffers"].items():
    assert id == len(root.buffers)
    buffer_t = tflite.BufferT()
    buffer_t.data = data
    root.buffers.append(buffer_t)

  size_hint = 1 * 2**20
  builder = flatbuffers.Builder(size_hint)
  builder.Finish(root.Pack(builder))
  flatbuffer = builder.Output()
  return flatbuffer


def get_buffer(spec: dict, subgraph: int, tensor: int) -> bytearray:
  """Return the buffer for a given tensor in a model spec.
  """
  tensor_spec = spec["subgraphs"][subgraph]["tensors"][tensor]
  buffer_id = tensor_spec["buffer"]
  buffer = spec["buffers"][buffer_id]
  return buffer


EXAMPLE_MODEL = {
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
                    "type": tflite.TensorType.INT8,
                    "buffer": 1,
                },
                1: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 1,
                },
                2: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 1,
                },
                3: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 1,
                },
            },
        },
    },
    "buffers": {
        0: bytes(),
        1: bytes((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
        2: bytes((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
        3: bytes((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
        4: bytes((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
    }
}
