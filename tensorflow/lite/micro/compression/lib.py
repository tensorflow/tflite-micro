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

import struct
import typing

from tensorflow.lite.python import schema_py_generated as tflite

import model_facade

METADATA_KEY = "COMPRESSION_METADATA"

# Operator input indices for which LUT-compression is implemented, by opcode
LUT_COMPRESSABLE_INPUTS = {
    tflite.BuiltinOperator.FULLY_CONNECTED: (1, 2),
}


def lut_compressable_inputs(
    model: model_facade.Model, ) -> typing.Sequence[model_facade.Tensor]:
  """LUT-compressable input tensors in the model."""

  tensors = set()
  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      indices = LUT_COMPRESSABLE_INPUTS.get(op.opcode.builtinCode, ())
      for i in indices:
        tensors.add(op.inputs[i])
  return tensors


_struct_formats = {
    tflite.TensorType.FLOAT32: "f",
    tflite.TensorType.FLOAT16: "e",
    tflite.TensorType.FLOAT64: "d",
    tflite.TensorType.INT8: "b",
    tflite.TensorType.INT16: "h",
    tflite.TensorType.INT32: "i",
    tflite.TensorType.INT64: "q",
    tflite.TensorType.UINT8: "B",
    tflite.TensorType.UINT16: "H",
    tflite.TensorType.UINT32: "I",
    tflite.TensorType.UINT64: "Q",
    tflite.TensorType.BOOL: "?",
}


def buffer_values(buffer: model_facade.Buffer, type_: tflite.TensorType):
  """Return properly-typed values unpacked from the given buffer.
  """
  little_endian = "<"  # always, per tflite schema
  format = little_endian + _struct_formats[type_]
  # iter_unpack yields tuples of length 1, unpack the tuples
  return [t[0] for t in struct.iter_unpack(format, buffer.data)]


def tensor_values(tensor: model_facade.Tensor):
  """Return properly-typed values for the given tensor.
  """
  return buffer_values(tensor.data, tensor.type)
