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
"""Tools for constructing flatbuffers for testing.

This module provides tools for constructing .tflite flatbuffers from a Python
dictionary representation of a model, a prototype of which can be found in
EXAMPLE_MODEL.

Example usage:
  model_definition = {...}  # use EXAMPLE_MODEL as prototype
  flatbuffer: bytearray = test_models.build(model_definition)
"""

# This module must remain low-level and independent from any helpers in this
# project which make constructing model and flatbuffers easier, because this
# module is used to define tests for those helpers.

import flatbuffers
import numpy as np
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

EXAMPLE_MODEL = {
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
                    "quantization": {
                        "quantized_dimension": 0,
                    },
                },
            },
        },
    },
    "buffers": {
        0: None,
        1: np.array(range(16), dtype=np.dtype("<i1")),
        2: np.array(range(16), dtype=np.dtype("<i1")),
        3: np.array(range(16), dtype=np.dtype("<i1")),
        4: np.array(range(16), dtype=np.dtype("<i1")),
    }
}


def build(model_definition: dict) -> bytearray:
  """Builds a .tflite flatbuffer from a model definition.

  Args:
    model_definition: A dictionary representation of the model, a prototype of
      which can be found in the EXAMPLE_MODEL attribute of this module.

  Returns:
    A tflite flatbuffer.
  """
  root = tflite.ModelT()
  description = model_definition.get("description")
  if description is not None:
    root.description = description

  root.operatorCodes = []
  for id, operator_code in model_definition["operator_codes"].items():
    assert id == len(root.operatorCodes)
    opcode_t = tflite.OperatorCodeT()
    root.operatorCodes.append(opcode_t)
    opcode_t.builtinCode = operator_code["builtin_code"]

  root.metadata = []
  if "metadata" in model_definition:
    for _, metadata in model_definition["metadata"].items():
      metadata_t = tflite.MetadataT()
      metadata_t.name = metadata["name"]
      metadata_t.buffer = metadata["buffer"]
      root.metadata.append(metadata_t)

  root.subgraphs = []
  for id, subgraph in model_definition["subgraphs"].items():
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
      tensor_t.name = tensor.get("name", None)
      tensor_t.shape = tensor["shape"]
      tensor_t.type = tensor["type"]
      tensor_t.buffer = tensor["buffer"]

      if "quantization" in tensor:
        tensor_t.quantization = tflite.QuantizationParametersT()
        tensor_t.quantization.quantizedDimension = \
            tensor["quantization"].get("quantized_dimension", None)
        tensor_t.quantization.scale = \
            tensor["quantization"].get("scale", None)
        tensor_t.quantization.zeroPoint = \
            tensor["quantization"].get("zero_point", None)

      subgraph_t.tensors.append(tensor_t)

  root.buffers = []
  for id, data in model_definition["buffers"].items():
    assert id == len(root.buffers)
    buffer_t = tflite.BufferT()

    if data is None:
      buffer_t.data = []
    elif isinstance(data, np.ndarray):
      array = data.astype(data.dtype.newbyteorder("<"))  # ensure little-endian
      buffer_t.data = list(array.tobytes())
    else:
      raise TypeError(f"buffer_id {id} must be None or an np.ndarray")

    root.buffers.append(buffer_t)

  size_hint = 1 * 2**20
  builder = flatbuffers.Builder(size_hint)
  builder.Finish(root.Pack(builder))
  flatbuffer = builder.Output()
  return flatbuffer
