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
"""Unified TFLite model manipulation module.

Provides a clean API for creating, reading, and modifying TFLite models.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np
import flatbuffers
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


class _BufferList(list):
  """Custom list that auto-sets buffer.index on append.

  When a buffer is appended, automatically sets buffer.index to its position.
  This enables append-only workflows to work seamlessly.
  """

  def append(self, buf):
    """Append buffer and auto-set its index."""
    buf.index = len(self)
    super().append(buf)


@dataclass
class Buffer:
  """Buffer holding tensor data.

  The index field indicates the buffer's position in the model's buffer array.
  It is automatically populated during:
  - read(): Set from flatbuffer
  - build(): Set during compilation
  - model.buffers.append(): Auto-set to len(model.buffers) - 1

  The index may become stale after:
  - Deleting buffers from model.buffers
  - Reordering buffers in model.buffers

  For append-only workflows (the common case), buffer.index can be trusted.
  """
  data: bytes
  index: Optional[int] = None

  def __len__(self):
    return len(self.data)

  def __bytes__(self):
    return self.data


@dataclass
class Quantization:
  """Quantization parameters helper."""
  scales: Union[float, List[float]]
  zero_points: Union[int, List[int]] = 0
  axis: Optional[int] = None

  def to_tflite(self) -> tflite.QuantizationParametersT:
    """Convert to TFLite schema object."""
    q = tflite.QuantizationParametersT()

    # Normalize to lists
    scales = [self.scales] if isinstance(self.scales,
                                         (int, float)) else self.scales
    zeros = [self.zero_points] if isinstance(self.zero_points,
                                             int) else self.zero_points

    q.scale = scales
    q.zeroPoint = zeros
    if self.axis is not None:
      q.quantizedDimension = self.axis

    return q


@dataclass
class Tensor:
  """Declarative tensor specification.

    Supports both buffer= and data= parameters for flexibility:
    - buffer=: Explicitly provide a Buffer object (can be shared between tensors)
    - data=: Convenience parameter that auto-creates a Buffer

    Cannot specify both buffer and data at initialization.
    """
  shape: tuple
  dtype: tflite.TensorType
  buffer: Optional[Buffer] = None
  quantization: Optional[Quantization] = None
  name: Optional[str] = None

  # Internal field for data initialization only
  _data_init: Optional[Union[bytes, np.ndarray]] = field(default=None,
                                                         init=False,
                                                         repr=False)

  # Auto-populated during build/read
  _index: Optional[int] = field(default=None, init=False, repr=False)

  def __init__(self,
               shape,
               dtype,
               buffer=None,
               data=None,
               quantization=None,
               name=None):
    """Initialize Tensor.

        Args:
            shape: Tensor shape as tuple
            dtype: TensorType enum value
            buffer: Optional Buffer object (for explicit buffer sharing)
            data: Optional numpy array or bytes (convenience parameter, creates Buffer)
            quantization: Optional Quantization object
            name: Optional tensor name

        Raises:
            ValueError: If both buffer and data are specified
        """
    if data is not None and buffer is not None:
      raise ValueError("Cannot specify both data and buffer")

    self.shape = shape
    self.dtype = dtype
    self.buffer = buffer
    self.quantization = quantization
    self.name = name
    self._index = None

    # Convert data to buffer if provided
    if data is not None:
      buf_data = data if isinstance(data, bytes) else data.tobytes()
      self.buffer = Buffer(data=buf_data)

  @property
  def array(self) -> Optional[np.ndarray]:
    """Get tensor data as properly-shaped numpy array.

        Returns:
            numpy array with shape matching tensor.shape and dtype matching
            tensor.dtype, or None if tensor has no data.

        For low-level byte access, use tensor.buffer.data instead.
        """
    if self.buffer is None:
      return None
    return np.frombuffer(self.buffer.data,
                         dtype=_dtype_to_numpy(self.dtype)).reshape(self.shape)

  @array.setter
  def array(self, value: np.ndarray):
    """Set tensor data from numpy array.

        Args:
            value: New tensor data as numpy array. Will be converted to bytes
                   using tobytes() and stored in the buffer.

        Creates a new Buffer if tensor has no buffer, or updates the existing
        buffer's data in place.

        For low-level byte access, use tensor.buffer.data instead.
        """
    buf_data = value.tobytes()
    if self.buffer is None:
      self.buffer = Buffer(data=buf_data)
    else:
      self.buffer.data = buf_data

  @property
  def index(self) -> Optional[int]:
    """Tensor index in the subgraph's tensor list.

        Returns index after read() or build(). May be None or stale after
        modifications. Use with caution.
        """
    return self._index

  @property
  def numpy_dtype(self) -> np.dtype:
    """Get numpy dtype corresponding to tensor's TFLite dtype.

        Returns:
            numpy dtype object for use with np.frombuffer, np.array, etc.
        """
    return _dtype_to_numpy(self.dtype)


@dataclass
class OperatorCode:
  """Operator code specification."""
  builtin_code: tflite.BuiltinOperator
  custom_code: Optional[str] = None
  version: int = 1


@dataclass
class Operator:
  """Declarative operator specification."""
  opcode: Union[tflite.BuiltinOperator, int]
  inputs: List[Tensor]
  outputs: List[Tensor]
  custom_code: Optional[str] = None

  # Set when reading from existing model
  opcode_index: Optional[int] = None

  _index: Optional[int] = field(default=None, init=False, repr=False)


@dataclass
class Subgraph:
  """Declarative subgraph specification with imperative methods."""
  tensors: List[Tensor] = field(default_factory=list)
  operators: List[Operator] = field(default_factory=list)
  inputs: List[Tensor] = field(default_factory=list)
  outputs: List[Tensor] = field(default_factory=list)
  name: Optional[str] = None

  _index: Optional[int] = field(default=None, init=False, repr=False)

  def add_tensor(self, **kwargs) -> Tensor:
    """Add tensor imperatively and return it."""
    t = Tensor(**kwargs)
    t._index = len(self.tensors)
    self.tensors.append(t)
    return t

  def add_operator(self, **kwargs) -> Operator:
    """Add operator imperatively and return it."""
    op = Operator(**kwargs)
    op._index = len(self.operators)
    self.operators.append(op)
    return op

  @property
  def index(self) -> Optional[int]:
    """Subgraph index in the model's subgraph list.

        Returns index after read() or build(). May be None or stale after
        modifications. Use with caution.
        """
    return self._index


@dataclass
class Model:
  """Top-level model specification."""
  subgraphs: List[Subgraph] = field(default_factory=list)
  buffers: _BufferList = field(
      default_factory=_BufferList)  # Auto-sets buffer.index on append
  operator_codes: List[OperatorCode] = field(default_factory=list)
  metadata: dict = field(default_factory=dict)
  description: Optional[str] = None

  def add_subgraph(self, **kwargs) -> Subgraph:
    """Add subgraph imperatively and return it."""
    sg = Subgraph(**kwargs)
    sg._index = len(self.subgraphs)
    self.subgraphs.append(sg)
    return sg

  def build(self) -> bytearray:
    """Compile to flatbuffer with automatic bookkeeping."""
    compiler = _ModelCompiler(self)
    return compiler.compile()


def read(buffer: bytes) -> Model:
  """Read a TFLite flatbuffer and return a Model object."""
  fb_model = tflite.ModelT.InitFromPackedBuf(buffer, 0)

  # Create Model with basic fields
  # Decode bytes to strings where needed
  description = fb_model.description
  if isinstance(description, bytes):
    description = description.decode('utf-8')

  model = Model(description=description)

  # Create all buffers first (so tensors can reference them)
  for i, fb_buf in enumerate(fb_model.buffers):
    buf_data = bytes(fb_buf.data) if fb_buf.data is not None else b''
    buf = Buffer(data=buf_data, index=i)
    model.buffers.append(buf)

  # Read operator codes
  for fb_opcode in fb_model.operatorCodes:
    custom_code = fb_opcode.customCode
    if isinstance(custom_code, bytes):
      custom_code = custom_code.decode('utf-8')

    opcode = OperatorCode(
        builtin_code=fb_opcode.builtinCode,
        custom_code=custom_code,
        version=fb_opcode.version if fb_opcode.version else 1)
    model.operator_codes.append(opcode)

  # Read subgraphs
  for sg_idx, fb_sg in enumerate(fb_model.subgraphs):
    sg = Subgraph()
    sg._index = sg_idx

    # Read tensors
    for tensor_idx, fb_tensor in enumerate(fb_sg.tensors):
      # Decode tensor name
      name = fb_tensor.name
      if isinstance(name, bytes):
        name = name.decode('utf-8')

      # Create tensor referencing the appropriate buffer
      # Buffer 0 is the empty buffer (TFLite convention), so treat it as None
      buf = None if fb_tensor.buffer == 0 else model.buffers[fb_tensor.buffer]

      # Read quantization parameters if present
      quant = None
      if fb_tensor.quantization:
        fb_quant = fb_tensor.quantization
        if fb_quant.scale is not None and len(fb_quant.scale) > 0:
          # Quantization parameters present
          scales = list(fb_quant.scale)
          zeros = list(
              fb_quant.zeroPoint
          ) if fb_quant.zeroPoint is not None else [0] * len(scales)

          # Handle axis: only set if per-channel (more than one scale)
          axis = None
          if len(scales) > 1 and fb_quant.quantizedDimension is not None:
            axis = fb_quant.quantizedDimension

          quant = Quantization(scales=scales, zero_points=zeros, axis=axis)

      shape = tuple(fb_tensor.shape) if fb_tensor.shape is not None else ()
      tensor = Tensor(shape=shape,
                      dtype=fb_tensor.type,
                      buffer=buf,
                      name=name,
                      quantization=quant)
      tensor._index = tensor_idx

      sg.tensors.append(tensor)

    # Read operators
    for fb_op in fb_sg.operators:
      # Get operator code info
      opcode_obj = model.operator_codes[fb_op.opcodeIndex]

      inputs = [sg.tensors[i] for i in fb_op.inputs] if fb_op.inputs is not None else []
      outputs = [sg.tensors[i] for i in fb_op.outputs] if fb_op.outputs is not None else []
      op = Operator(opcode=opcode_obj.builtin_code,
                    inputs=inputs,
                    outputs=outputs,
                    custom_code=opcode_obj.custom_code,
                    opcode_index=fb_op.opcodeIndex)
      sg.operators.append(op)

    # Read subgraph inputs/outputs
    if fb_sg.inputs is not None and len(fb_sg.inputs) > 0:
      sg.inputs = [sg.tensors[i] for i in fb_sg.inputs]
    if fb_sg.outputs is not None and len(fb_sg.outputs) > 0:
      sg.outputs = [sg.tensors[i] for i in fb_sg.outputs]

    model.subgraphs.append(sg)

  # Read metadata
  if fb_model.metadata:
    for entry in fb_model.metadata:
      # Decode metadata name
      name = entry.name
      if isinstance(name, bytes):
        name = name.decode('utf-8')

      # Get metadata value from buffer
      buffer = fb_model.buffers[entry.buffer]
      value = bytes(buffer.data) if buffer.data is not None else b''

      model.metadata[name] = value

  return model


def _dtype_to_numpy(dtype: tflite.TensorType) -> np.dtype:
  """Convert TFLite dtype to numpy dtype."""
  type_map = {
      tflite.TensorType.INT8: np.int8,
      tflite.TensorType.INT16: np.int16,
      tflite.TensorType.INT32: np.int32,
      tflite.TensorType.UINT8: np.uint8,
      tflite.TensorType.FLOAT32: np.float32,
  }
  return type_map.get(dtype, np.uint8)


class _ModelCompiler:
  """Internal: compiles Model to flatbuffer with automatic bookkeeping."""

  def __init__(self, model: Model):
    self.model = model
    self._buffers = []
    self._buffer_map = {}  # Map Buffer object id to index
    self._operator_codes = {}

  def compile(self) -> bytearray:
    """Compile model to flatbuffer."""
    root = tflite.ModelT()
    root.version = 3

    # Set description
    root.description = self.model.description

    # Initialize buffers
    # If model.buffers exists (from read()), preserve those buffers
    if self.model.buffers:
      for buf in self.model.buffers:
        fb_buf = tflite.BufferT()
        fb_buf.data = list(buf.data) if buf.data else []
        self._buffers.append(fb_buf)
        self._buffer_map[id(buf)] = buf.index
    else:
      # Creating model from scratch: initialize buffer 0 as empty (TFLite convention)
      empty_buffer = tflite.BufferT()
      empty_buffer.data = []
      self._buffers = [empty_buffer]
      # Note: buffer 0 should not be in _buffer_map since tensors without data use it

    # Auto-collect and register operator codes
    self._collect_operator_codes()
    root.operatorCodes = list(self._operator_codes.values())

    # Process subgraphs
    root.subgraphs = []
    for sg in self.model.subgraphs:
      root.subgraphs.append(self._compile_subgraph(sg))

    # Process buffers
    root.buffers = self._buffers

    # Process metadata
    root.metadata = self._compile_metadata()

    # Pack and return
    builder = flatbuffers.Builder(4 * 2**20)
    builder.Finish(root.Pack(builder))
    return builder.Output()

  def _collect_operator_codes(self):
    """Scan all operators and build operator code table."""
    for sg in self.model.subgraphs:
      for op in sg.operators:
        key = (op.opcode, op.custom_code)
        if key not in self._operator_codes:
          opcode = tflite.OperatorCodeT()
          opcode.builtinCode = op.opcode
          if op.custom_code:
            opcode.customCode = op.custom_code
          self._operator_codes[key] = opcode

  def _compile_subgraph(self, sg: Subgraph) -> tflite.SubGraphT:
    """Compile subgraph, extracting inline tensors from operators."""
    sg_t = tflite.SubGraphT()
    sg_t.name = sg.name

    # Collect all tensors (from tensor list and inline in operators)
    all_tensors = list(sg.tensors)
    tensor_to_index = {}
    for i, t in enumerate(all_tensors):
      t._index = i
      tensor_to_index[id(t)] = i

    # Extract inline tensors from operators and subgraph inputs/outputs
    inline_sources = [op.inputs + op.outputs for op in sg.operators]
    inline_sources.append(sg.inputs)
    inline_sources.append(sg.outputs)
    for source in inline_sources:
      for tensor in source:
        if id(tensor) not in tensor_to_index:
          tensor._index = len(all_tensors)
          tensor_to_index[id(tensor)] = tensor._index
          all_tensors.append(tensor)

    # Compile all tensors
    sg_t.tensors = []
    for tensor in all_tensors:
      sg_t.tensors.append(self._compile_tensor(tensor))

    # Compile operators
    sg_t.operators = []
    for op in sg.operators:
      sg_t.operators.append(self._compile_operator(op, tensor_to_index))

    # Set subgraph inputs/outputs
    sg_t.inputs = [tensor_to_index[id(t)] for t in sg.inputs]
    sg_t.outputs = [tensor_to_index[id(t)] for t in sg.outputs]

    return sg_t

  def _compile_operator(self, op: Operator,
                        tensor_to_index: dict) -> tflite.OperatorT:
    """Compile operator, resolving tensor references and opcodes."""
    op_t = tflite.OperatorT()

    # Get opcode index
    key = (op.opcode, op.custom_code)
    opcode_index = list(self._operator_codes.keys()).index(key)
    op_t.opcodeIndex = opcode_index

    # Resolve tensor references to indices
    op_t.inputs = [tensor_to_index[id(inp)] for inp in op.inputs]
    op_t.outputs = [tensor_to_index[id(outp)] for outp in op.outputs]

    return op_t

  def _compile_tensor(self, tensor: Tensor) -> tflite.TensorT:
    """Compile tensor, reusing or creating buffer as needed."""
    t = tflite.TensorT()
    t.shape = list(tensor.shape)
    t.type = tensor.dtype
    t.name = tensor.name

    # Handle buffer assignment
    if tensor.buffer is None:
      # No data: use buffer 0
      t.buffer = 0
    else:
      # Has buffer: get or create index for it
      buf_id = id(tensor.buffer)
      if buf_id not in self._buffer_map:
        # First time seeing this buffer, add it
        fb_buf = tflite.BufferT()
        fb_buf.data = list(tensor.buffer.data)
        self._buffers.append(fb_buf)
        buf_index = len(self._buffers) - 1
        self._buffer_map[buf_id] = buf_index
        tensor.buffer.index = buf_index
      t.buffer = self._buffer_map[buf_id]

    # Handle quantization
    if tensor.quantization:
      t.quantization = tensor.quantization.to_tflite()

    return t

  def _compile_metadata(self):
    """Compile metadata, creating buffers for metadata values."""
    if not self.model.metadata:
      return []

    metadata_entries = []
    for name, value in self.model.metadata.items():
      # Create buffer for metadata value
      buf = tflite.BufferT()
      buf.data = list(value) if isinstance(value, bytes) else list(value)
      self._buffers.append(buf)
      buf_index = len(self._buffers) - 1

      # Create metadata entry
      entry = tflite.MetadataT()
      entry.name = name
      entry.buffer = buf_index
      metadata_entries.append(entry)

    return metadata_entries
