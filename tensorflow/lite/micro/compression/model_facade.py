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
"""A facade for working with tflite.Model.

This module provides convenient navigation, data type conversions, and
utilities for working with a tflite.Model, which can be tedious and verbose to
work with directly.

Usage:
  model = model_facade.read(flatbuffer)
  # manipulate
  new_flatbuffer = model.compile()
"""

from __future__ import annotations

import flatbuffers
import numpy as np
from numpy.typing import NDArray
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite
from typing import ByteString, Generic, TypeVar

_IteratorTo = TypeVar("_IteratorTo")


class _Iterator(Generic[_IteratorTo]):

  def __init__(self, sequence, cls, parent):
    self._sequence = sequence
    self._cls = cls
    self._index = 0
    self._parent = parent

  def __getitem__(self, key) -> _IteratorTo:
    return self._cls(self._sequence[key], key, self._parent)

  def __len__(self):
    return len(self._sequence)

  def __iter__(self):
    self._index = 0
    return self

  def __next__(self):
    try:
      result = self[self._index]
      self._index += 1
      return result
    except IndexError:
      raise StopIteration


class _IndirectIterator(Generic[_IteratorTo]):

  def __init__(self, indices, sequence):
    self._indices = indices
    self._index = 0
    self._sequence = sequence

  def __getitem__(self, key) -> _IteratorTo:
    index = self._indices[key]
    return self._sequence[index]

  def __len__(self):
    return len(self._indices)

  def __iter__(self):
    self._index = 0
    return self

  def __next__(self):
    try:
      result = self[self._index]
      self._index += 1
      return result
    except IndexError:
      raise StopIteration


class _Operator:

  def __init__(self, operator, index, subgraph):
    self.operator = operator
    self.index = index
    self.subgraph = subgraph

  @property
  def opcode(self) -> tflite.OperatorCodeT:
    return self.subgraph.model.operatorCodes[self.operator.opcodeIndex]

  @property
  def inputs(self):
    return _IndirectIterator(self.operator.inputs, self.subgraph.tensors)


_NP_DTYPES = {
    tflite.TensorType.FLOAT16: np.dtype("<f2"),
    tflite.TensorType.FLOAT32: np.dtype("<f4"),
    tflite.TensorType.FLOAT64: np.dtype("<f8"),
    tflite.TensorType.INT8: np.dtype("<i1"),
    tflite.TensorType.INT16: np.dtype("<i2"),
    tflite.TensorType.INT32: np.dtype("<i4"),
    tflite.TensorType.INT64: np.dtype("<i8"),
    tflite.TensorType.UINT8: np.dtype("<u1"),
    tflite.TensorType.UINT16: np.dtype("<u2"),
    tflite.TensorType.UINT32: np.dtype("<u4"),
    tflite.TensorType.UINT64: np.dtype("<u8"),
}


class _Tensor:

  def __init__(self, tensor_t: tflite.TensorT, index, subgraph: _Subgraph):
    self._tensor_t = tensor_t
    self.index = index
    self.subgraph = subgraph

  @property
  def name(self):
    n = self._tensor_t.name
    if isinstance(n, bytes):
      return n.decode("utf-8")
    else:
      return n

  @property
  def shape(self):
    """Return the shape as specified in the model.
    """
    return self._tensor_t.shape

  @property
  def buffer_index(self):
    return self._tensor_t.buffer

  @property
  def buffer(self) -> _Buffer:
    return self.subgraph.model.buffers[self._tensor_t.buffer]

  @property
  def data(self) -> bytes:
    return self.buffer.data

  @property
  def dtype(self) -> np.dtype:
    return _NP_DTYPES[self._tensor_t.type]

  @property
  def array(self) -> np.ndarray:
    """Returns an array created from the Tensor's data, type, and shape.

    Note the bytes in the data buffer and the Tensor's type and shape may be
    inconsistent, and thus the returned array invalid, if the data buffer has
    been altered according to the compression schema, in which the data buffer
    is an array of fixed-width, integer fields.
    """
    return np.frombuffer(self.data,
                         dtype=self.dtype).reshape(self._tensor_t.shape)

  @property
  def quantization(self) -> tflite.QuantizationParametersT | None:
    return self._tensor_t.quantization


class _Buffer:

  def __init__(self, buffer_t: tflite.BufferT, index, model):
    self._buffer_t = buffer_t
    self.index = index
    self.model = model

  @property
  def data(self) -> bytes:
    return bytes(self._buffer_t.data)

  @data.setter
  def data(self, value: ByteString):
    self._buffer_t.data = list(value)

  def extend(self, values: NDArray):
    self._buffer_t.data.extend(values.tobytes())


class _Subgraph:

  def __init__(self, subgraph_t: tflite.SubGraphT, index: int, model: _Model):
    self._subgraph_t = subgraph_t
    self.index = index
    self.model = model

  @property
  def operators(self) -> _Iterator[_Operator]:
    return _Iterator(self._subgraph_t.operators, _Operator, parent=self)

  @property
  def tensors(self) -> _Iterator[_Tensor]:
    return _Iterator(self._subgraph_t.tensors, _Tensor, parent=self)


class _Model:
  """A facade for manipulating tflite.Model.
  """

  def __init__(self, model_t: tflite.ModelT):
    self._model_t = model_t

  def compile(self) -> bytearray:
    """Returns a tflite.Model flatbuffer.
    """
    size_hint = 4 * 2**10
    builder = flatbuffers.Builder(size_hint)
    builder.Finish(self._model_t.Pack(builder))
    return builder.Output()

  def add_buffer(self) -> _Buffer:
    """Adds a buffer to the model.
    """
    buffer = tflite.BufferT()
    buffer.data = []
    self._model_t.buffers.append(buffer)
    index = len(self._model_t.buffers) - 1
    return _Buffer(buffer, index, self._model_t)

  def add_metadata(self, key, value):
    """Adds a key-value pair, writing value to a newly created buffer.
    """
    metadata = tflite.MetadataT()
    metadata.name = key
    buffer = self.add_buffer()
    buffer.data = value
    metadata.buffer = buffer.index
    self._model_t.metadata.append(metadata)

  @property
  def metadata(self) -> dict[str, _Buffer]:
    """Returns the model's metadata as a dictionary to Buffer objects.
    """
    result = {}
    for m in self._model_t.metadata:
      name = m.name.decode("utf-8")  # type: ignore (fb library is wrong)
      buffer = _Buffer(self._model_t.buffers[m.buffer], m.buffer,
                       self._model_t)
      result[name] = buffer

    return result

  @property
  def operatorCodes(self):
    return self._model_t.operatorCodes

  @property
  def subgraphs(self) -> _Iterator[_Subgraph]:
    return _Iterator(self._model_t.subgraphs, _Subgraph, parent=self)

  @property
  def buffers(self) -> _Iterator[_Buffer]:
    return _Iterator(self._model_t.buffers, _Buffer, parent=self)


def read(buffer: ByteString):
  """Reads a tflite.Model and returns a model facade.
  """
  schema_model = tflite.ModelT.InitFromPackedBuf(buffer, 0)
  return _Model(schema_model)
