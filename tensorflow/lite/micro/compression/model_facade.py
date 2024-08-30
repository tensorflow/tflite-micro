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
"""A facade for manipulating tflite.Model.

Usage:
  model = model_facade.read_file(path)
  # manipulate model
  model.write_file(path)

A tflite.Model can be tedious and verbose to navigate.
"""

# TODO: make a better distinction between object representation objects
# and facade objects.

from typing import Sequence

from tensorflow.lite.python import schema_py_generated as tflite

import flatbuffers
import struct


def read(buffer: bytes):
  """Read a tflite.Model from a buffer and return a model facade."""
  schema_model = tflite.ModelT.InitFromPackedBuf(buffer, 0)
  return Model(schema_model)


class Model:
  """A facade for manipulating tflite.Model."""

  def __init__(self, representation: tflite.ModelT):
    self.root = representation

  def pack(self) -> bytearray:
    """Pack and return the tflite.Model as a flatbuffer."""
    size_hint = 4 * 2**10
    builder = flatbuffers.Builder(size_hint)
    builder.Finish(self.root.Pack(builder))
    return builder.Output()

  def add_buffer(self):
    """Add a buffer to the model and return a Buffer facade."""
    buffer = tflite.BufferT()
    buffer.data = []
    self.root.buffers.append(buffer)
    index = len(self.root.buffers) - 1
    return Buffer(buffer, index, self.root)

  def add_metadata(self, key, value):
    """Add a key-value pair, writing value to newly created tflite.Buffer."""
    metadata = tflite.MetadataT()
    metadata.name = key
    buffer = self.add_buffer()
    buffer.data = value
    metadata.buffer = buffer.index
    self.root.metadata.append(metadata)

  @property
  def operatorCodes(self):
    return self.root.operatorCodes

  @property
  def subgraphs(self):
    return Iterator(self.root.subgraphs, Subgraph, parent=self)

  @property
  def buffers(self):
    return Iterator(self.root.buffers, Buffer, parent=self)


class Iterator:

  def __init__(self, sequence, cls, parent):
    self._sequence = sequence
    self._cls = cls
    self._parent = parent

  def __getitem__(self, key):
    return self._cls(self._sequence[key], key, self._parent)

  def __len__(self):
    return len(self._sequence)


class IndirectIterator:

  def __init__(self, indices, sequence):
    self._indices = indices
    self._sequence = sequence

  def __getitem__(self, key):
    index = self._indices[key]
    return self._sequence[index]

  def __len__(self):
    return len(self._indices)


class Subgraph:

  def __init__(self, subgraph, index, model):
    self.subgraph = subgraph
    self.index = index
    self.model = model

  @property
  def operators(self):
    return Iterator(self.subgraph.operators, Operator, parent=self)

  @property
  def tensors(self):
    return Iterator(self.subgraph.tensors, Tensor, parent=self)


class Operator:

  def __init__(self, operator, index, subgraph):
    self.operator = operator
    self.index = index
    self.subgraph = subgraph

  @property
  def opcode(self) -> tflite.OperatorCodeT:
    return self.subgraph.model.operatorCodes[self.operator.opcodeIndex]

  @property
  def inputs(self):
    return IndirectIterator(self.operator.inputs, self.subgraph.tensors)


class Tensor:

  def __init__(self, tensor, index, subgraph):
    self.tensor = tensor
    self.index = index
    self.subgraph = subgraph

  @property
  def name(self):
    return self.tensor.name.decode('utf-8')

  @property
  def shape(self):
    """Return the shape as specified in the model.
    """
    return self.tensor.shape

  @property
  def shape_nhwc(self):
    """Return the shape normalized to a full (N,H,W,C) vector.
    """
    n_missing_dims = 4 - len(self.tensor.shape)
    nhwc = [1 for _ in range(0, n_missing_dims)]
    nhwc.extend(self.tensor.shape)
    return nhwc

  @property
  def buffer(self):
    return self.subgraph.model.buffers[self.tensor.buffer]

  @property
  def data(self):
    return self.buffer.data

  @property
  def type(self):
    return self.tensor.type

  @property
  def values(self):
    reader = struct.iter_unpack(_struct_formats[self.type], self.data)
    # iter_unpack yields tuples of length 1, unpack the tuples
    return [value[0] for value in reader]

  @property
  def channel_count(self):
    if (self.tensor.quantization is None or
        self.tensor.quantization.scale is None):
      return 1
    return len(self.tensor.quantization.scale)


_struct_formats = {
    tflite.TensorType.FLOAT32: "<f",
    tflite.TensorType.FLOAT16: "<e",
    tflite.TensorType.FLOAT64: "<d",
    tflite.TensorType.INT8: "<b",
    tflite.TensorType.INT16: "<h",
    tflite.TensorType.INT32: "<i",
    tflite.TensorType.INT64: "<q",
    tflite.TensorType.UINT8: "<B",
    tflite.TensorType.UINT16: "<H",
    tflite.TensorType.UINT32: "<I",
    tflite.TensorType.UINT64: "<Q",
    tflite.TensorType.BOOL: "<?",
}


class Buffer:

  def __init__(self, buffer, index, model):
    self.buffer = buffer
    self.index = index
    self.model = model

  @property
  def data(self):
    return self.buffer.data

  @data.setter
  def data(self, value):
    self.buffer.data = value
    return self.data

  def extend_values(self, values: Sequence, type: int):
    for v in values:
      octets = struct.pack(_struct_formats[type], v)
      self.buffer.data.extend(octets)
