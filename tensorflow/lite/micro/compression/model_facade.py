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

Provide convenient navigation and data types for working with tflite.Model,
which can be tedious and verbose to working with directly.

Usage:
  model = model_facade.read(flatbuffer)
  # manipulate
  new_flatbuffer = model.compile()
"""

import flatbuffers
import numpy as np
from numpy.typing import NDArray
from typing import ByteString, Generic, TypeVar

from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

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
  def builtin_opcode(self) -> int:
    result: int = self.opcode.deprecatedBuiltinCode
    if result == tflite.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
      result = self.opcode.builtinCode
    return result

  @property
  def inputs(self):
    return _IndirectIterator(self.operator.inputs, self.subgraph.tensors)

  @property
  def outputs(self):
    return _IndirectIterator(self.operator.outputs, self.subgraph.tensors)

  @property
  def inputs_indices(self):
    return self.operator.inputs

  @property
  def outputs_indices(self):
    return self.operator.outputs

  @property
  def builtin_options_type(self) -> int:
    return self.operator.builtinOptionsType

  @property
  def builtin_options(self):
    return self.operator.builtinOptions


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

  def __init__(self, tensor: tflite.TensorT, index, subgraph):
    self._tensor = tensor
    self.index = index
    self.subgraph = subgraph

  @property
  def name(self):
    n = self._tensor.name
    if isinstance(n, bytes):
      return n.decode("utf-8")
    else:
      return n

  @property
  def shape(self):
    """Return the shape as specified in the model.
    """
    return self._tensor.shape

  @property
  def buffer_index(self):
    return self._tensor.buffer

  @property
  def buffer(self):
    return self.subgraph.model.buffers[self._tensor.buffer]

  @property
  def data(self):
    return self.buffer.data

  @property
  def dtype(self) -> np.dtype:
    return _NP_DTYPES[self._tensor.type]

  @property
  def array(self) -> np.ndarray:
    """Returns an array created from the Tensor's data, type, and shape.

    Note the bytes in the data buffer and the Tensor's type and shape may be
    inconsistent, and thus the returned array invalid, if the data buffer has
    been altered according to the compression schema, in which the data buffer
    is an array of fixed-width, integer fields.
    """
    return np.frombuffer(self.data,
                         dtype=self.dtype).reshape(self._tensor.shape)

  @property
  def quantization(self):
    return self._tensor.quantization


class _Buffer:

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

  def extend(self, values: NDArray):
    self.buffer.data.extend(values.tobytes())


class _Subgraph:

  def __init__(self, subgraph, index, model):
    self.subgraph = subgraph
    self.index = index
    self.model = model

  @property
  def operators(self) -> _Iterator[_Operator]:
    return _Iterator(self.subgraph.operators, _Operator, parent=self)

  @property
  def tensors(self) -> _Iterator[_Tensor]:
    return _Iterator(self.subgraph.tensors, _Tensor, parent=self)


class _Model:
  """A facade for manipulating tflite.Model.
  """

  def __init__(self, representation: tflite.ModelT):
    self.root = representation

  def compile(self) -> bytearray:
    """Returns a tflite.Model flatbuffer.
    """
    size_hint = 4 * 2**10
    builder = flatbuffers.Builder(size_hint)
    builder.Finish(self.root.Pack(builder))
    return builder.Output()

  def add_buffer(self) -> _Buffer:
    """Adds a buffer to the model.
    """
    buffer = tflite.BufferT()
    buffer.data = []
    self.root.buffers.append(buffer)
    index = len(self.root.buffers) - 1
    return _Buffer(buffer, index, self.root)

  def add_metadata(self, key, value):
    """Adds a key-value pair, writing value to a newly created buffer.
    """
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
  def subgraphs(self) -> _Iterator[_Subgraph]:
    return _Iterator(self.root.subgraphs, _Subgraph, parent=self)

  @property
  def buffers(self) -> _Iterator[_Buffer]:
    return _Iterator(self.root.buffers, _Buffer, parent=self)


def read(buffer: ByteString):
  """Reads a tflite.Model and returns a model facade.
  """
  schema_model = tflite.ModelT.InitFromPackedBuf(buffer, 0)
  return _Model(schema_model)
