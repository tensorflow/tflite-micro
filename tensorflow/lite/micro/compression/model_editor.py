# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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


class Tensor:
  """Tensor specification wrapping a TensorT flatbuffer object.

  Provides clean APIs for common fields (shape, dtype, name, buffer,
  quantization) while preserving all other TensorT fields during
  read-modify-write.

  Supports both buffer= and data= parameters for flexibility:
  - buffer=: Explicitly provide a Buffer object (can be shared between tensors)
  - data=: Convenience parameter that auto-creates a Buffer

  Cannot specify both buffer and data at initialization.
  """

  def __init__(self,
               shape=None,
               dtype=None,
               buffer=None,
               data=None,
               quantization=None,
               name=None,
               _fb: tflite.TensorT = None):
    """Initialize Tensor.

    Args:
        shape: Tensor shape as tuple
        dtype: TensorType enum value
        buffer: Optional Buffer object (for explicit buffer sharing)
        data: Optional numpy array or bytes (convenience, creates Buffer)
        quantization: Optional Quantization object
        name: Optional tensor name
        _fb: Optional TensorT for wrapping existing flatbuffer object

    Raises:
        ValueError: If both buffer and data are specified
    """
    if data is not None and buffer is not None:
      raise ValueError("Cannot specify both data and buffer")

    # Use provided TensorT or create new one
    self._fb = _fb if _fb is not None else tflite.TensorT()
    self._index = None

    # Buffer object (managed separately; _fb.buffer is just an index)
    self.buffer = buffer

    # Quantization object (managed separately; synced to _fb on compile)
    self.quantization = quantization

    # Set fields if provided (these override any values in _fb)
    if shape is not None:
      self.shape = shape
    if dtype is not None:
      self.dtype = dtype
    if name is not None:
      self.name = name

    # Convert data to buffer if provided
    if data is not None:
      buf_data = data if isinstance(data, bytes) else data.tobytes()
      self.buffer = Buffer(data=buf_data)

  @property
  def shape(self) -> tuple:
    """Tensor shape as tuple."""
    return tuple(self._fb.shape) if self._fb.shape is not None else ()

  @shape.setter
  def shape(self, value):
    self._fb.shape = list(value)

  @property
  def dtype(self) -> tflite.TensorType:
    """Tensor data type."""
    return self._fb.type

  @dtype.setter
  def dtype(self, value: tflite.TensorType):
    self._fb.type = value

  @property
  def name(self) -> Optional[str]:
    """Tensor name for debugging."""
    n = self._fb.name
    if isinstance(n, bytes):
      return n.decode('utf-8')
    return n

  @name.setter
  def name(self, value: Optional[str]):
    self._fb.name = value

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


class OperatorCode:
  """Operator code specification wrapping an OperatorCodeT flatbuffer object.

  Provides clean APIs for common fields (builtin_code, custom_code, version)
  while preserving all other OperatorCodeT fields during read-modify-write.
  """

  def __init__(self,
               builtin_code: tflite.BuiltinOperator = None,
               custom_code: Optional[str] = None,
               version: int = 1,
               _fb: tflite.OperatorCodeT = None):
    """Initialize OperatorCode.

    Args:
        builtin_code: BuiltinOperator enum value
        custom_code: Custom operator name (for CUSTOM opcode)
        version: Operator version
        _fb: Optional OperatorCodeT for wrapping existing flatbuffer object
    """
    # Use provided OperatorCodeT or create new one
    self._fb = _fb if _fb is not None else tflite.OperatorCodeT()

    # Set fields if provided (these override any values in _fb)
    if builtin_code is not None:
      self.builtin_code = builtin_code
    if custom_code is not None:
      self.custom_code = custom_code
    if version != 1 or _fb is None:
      self.version = version

  @property
  def builtin_code(self) -> tflite.BuiltinOperator:
    """Builtin operator code."""
    return self._fb.builtinCode

  @builtin_code.setter
  def builtin_code(self, value: tflite.BuiltinOperator):
    self._fb.builtinCode = value

  @property
  def custom_code(self) -> Optional[str]:
    """Custom operator name (for CUSTOM opcode)."""
    c = self._fb.customCode
    if isinstance(c, bytes):
      return c.decode('utf-8')
    return c

  @custom_code.setter
  def custom_code(self, value: Optional[str]):
    self._fb.customCode = value

  @property
  def version(self) -> int:
    """Operator version."""
    return self._fb.version if self._fb.version else 1

  @version.setter
  def version(self, value: int):
    self._fb.version = value


class Operator:
  """Operator specification wrapping an OperatorT flatbuffer object.

  Provides clean APIs for common fields (opcode, inputs, outputs, custom_code)
  while preserving all other OperatorT fields (builtin_options, custom_options,
  intermediates, mutating_variable_inputs, etc.) during read-modify-write.
  """

  def __init__(self,
               opcode: Union[tflite.BuiltinOperator, int] = None,
               inputs: List[Tensor] = None,
               outputs: List[Tensor] = None,
               custom_code: Optional[str] = None,
               opcode_index: Optional[int] = None,
               _fb: tflite.OperatorT = None):
    """Initialize Operator.

    Args:
        opcode: BuiltinOperator enum value or CUSTOM
        inputs: List of input Tensor objects
        outputs: List of output Tensor objects
        custom_code: Custom operator name (for CUSTOM opcode)
        opcode_index: Index into operator_codes (set during read)
        _fb: Optional OperatorT for wrapping existing flatbuffer object
    """
    # Use provided OperatorT or create new one
    self._fb = _fb if _fb is not None else tflite.OperatorT()
    self._index = None

    # Tensor lists (managed separately; _fb stores indices, not objects)
    self.inputs = inputs if inputs is not None else []
    self.outputs = outputs if outputs is not None else []

    # These are derived from OperatorCode, not stored in OperatorT directly
    self._opcode = opcode
    self._custom_code = custom_code
    self._opcode_index = opcode_index

  @property
  def opcode(self) -> Union[tflite.BuiltinOperator, int]:
    """Builtin operator code."""
    return self._opcode

  @opcode.setter
  def opcode(self, value: Union[tflite.BuiltinOperator, int]):
    self._opcode = value

  @property
  def custom_code(self) -> Optional[str]:
    """Custom operator name (for CUSTOM opcode)."""
    return self._custom_code

  @custom_code.setter
  def custom_code(self, value: Optional[str]):
    self._custom_code = value

  @property
  def opcode_index(self) -> Optional[int]:
    """Index into operator_codes array (from read or after build)."""
    return self._opcode_index

  @opcode_index.setter
  def opcode_index(self, value: Optional[int]):
    self._opcode_index = value

  @property
  def index(self) -> Optional[int]:
    """Operator index in the subgraph's operator list."""
    return self._index


class Subgraph:
  """Subgraph specification wrapping a SubGraphT flatbuffer object.

  Provides clean APIs for common fields (tensors, operators, inputs, outputs,
  name) while preserving all other SubGraphT fields during read-modify-write.
  """

  def __init__(self,
               tensors: List[Tensor] = None,
               operators: List[Operator] = None,
               inputs: List[Tensor] = None,
               outputs: List[Tensor] = None,
               name: Optional[str] = None,
               _fb: tflite.SubGraphT = None):
    """Initialize Subgraph.

    Args:
        tensors: List of Tensor objects
        operators: List of Operator objects
        inputs: List of input Tensor objects
        outputs: List of output Tensor objects
        name: Subgraph name for debugging
        _fb: Optional SubGraphT for wrapping existing flatbuffer object
    """
    # Use provided SubGraphT or create new one
    self._fb = _fb if _fb is not None else tflite.SubGraphT()
    self._index = None

    # Lists of objects (managed separately; _fb stores indices/arrays)
    self.tensors = tensors if tensors is not None else []
    self.operators = operators if operators is not None else []
    self.inputs = inputs if inputs is not None else []
    self.outputs = outputs if outputs is not None else []

    # Set name if provided (overrides _fb value)
    if name is not None:
      self.name = name

  @property
  def name(self) -> Optional[str]:
    """Subgraph name for debugging."""
    n = self._fb.name
    if isinstance(n, bytes):
      return n.decode('utf-8')
    return n

  @name.setter
  def name(self, value: Optional[str]):
    self._fb.name = value

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

  def tensor_by_name(self, name: str) -> Tensor:
    """Look up a tensor by name.

    Args:
      name: The tensor name to find.

    Returns:
      The Tensor with the given name.

    Raises:
      KeyError: If no tensor with that name exists.
    """
    for t in self.tensors:
      if t.name == name:
        return t
    raise KeyError(f"No tensor named {name!r}")

  @property
  def index(self) -> Optional[int]:
    """Subgraph index in the model's subgraph list.

    Returns index after read() or build(). May be None or stale after
    modifications. Use with caution.
    """
    return self._index


class Model:
  """Model specification wrapping a ModelT flatbuffer object.

  Provides clean APIs for common fields (subgraphs, buffers, operator_codes,
  metadata, description) while preserving all other ModelT fields during
  read-modify-write.
  """

  def __init__(self,
               subgraphs: List[Subgraph] = None,
               buffers: _BufferList = None,
               operator_codes: List[OperatorCode] = None,
               metadata: dict = None,
               description: Optional[str] = None,
               _fb: tflite.ModelT = None):
    """Initialize Model.

    Args:
        subgraphs: List of Subgraph objects
        buffers: BufferList for tensor data
        operator_codes: List of OperatorCode objects
        metadata: Dict of metadata name -> bytes
        description: Model description string
        _fb: Optional ModelT for wrapping existing flatbuffer object
    """
    # Use provided ModelT or create new one
    self._fb = _fb if _fb is not None else tflite.ModelT()

    # Lists of objects (managed separately; _fb stores arrays)
    self.subgraphs = subgraphs if subgraphs is not None else []
    self.buffers = buffers if buffers is not None else _BufferList()
    self.operator_codes = operator_codes if operator_codes is not None else []
    self.metadata = metadata if metadata is not None else {}

    # Set description if provided (overrides _fb value)
    if description is not None:
      self.description = description

  @property
  def description(self) -> Optional[str]:
    """Model description string."""
    d = self._fb.description
    if isinstance(d, bytes):
      return d.decode('utf-8')
    return d

  @description.setter
  def description(self, value: Optional[str]):
    self._fb.description = value

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

  # Create Model wrapping the ModelT; all fields preserved in _fb
  model = Model(_fb=fb_model)

  # Create all buffers first (so tensors can reference them)
  for i, fb_buf in enumerate(fb_model.buffers):
    buf_data = bytes(fb_buf.data) if fb_buf.data is not None else b''
    buf = Buffer(data=buf_data, index=i)
    model.buffers.append(buf)

  # Read operator codes
  for fb_opcode in fb_model.operatorCodes:
    # Create OperatorCode wrapping the OperatorCodeT; all fields preserved in _fb
    opcode = OperatorCode(_fb=fb_opcode)
    model.operator_codes.append(opcode)

  # Read subgraphs
  for sg_idx, fb_sg in enumerate(fb_model.subgraphs):
    # Create Subgraph wrapping the SubGraphT; all fields preserved in _fb
    sg = Subgraph(_fb=fb_sg)
    sg._index = sg_idx

    # Read tensors
    for tensor_idx, fb_tensor in enumerate(fb_sg.tensors):
      # Resolve buffer reference
      # Buffer 0 is the empty buffer (TFLite convention), so treat it as None
      buf = None if fb_tensor.buffer == 0 else model.buffers[fb_tensor.buffer]

      # Read quantization parameters if present
      quant = None
      if fb_tensor.quantization:
        fb_quant = fb_tensor.quantization
        if fb_quant.scale is not None and len(fb_quant.scale) > 0:
          scales = list(fb_quant.scale)
          # Copy zero_points as-is, don't expand (per review feedback)
          zeros = list(
              fb_quant.zeroPoint) if fb_quant.zeroPoint is not None else [0]
          # Copy axis if: (1) it's non-zero, or (2) there are multiple scales.
          # This preserves per-channel quant with 1 channel (axis non-zero, 1 scale)
          # while treating default axis=0 with 1 scale as per-tensor (axis=None).
          axis = fb_quant.quantizedDimension
          if axis == 0 and len(scales) == 1:
            axis = None
          quant = Quantization(scales=scales, zero_points=zeros, axis=axis)

      # Create Tensor wrapping the TensorT; all fields preserved in _fb
      tensor = Tensor(_fb=fb_tensor, buffer=buf, quantization=quant)
      tensor._index = tensor_idx

      sg.tensors.append(tensor)

    # Read operators
    for fb_op in fb_sg.operators:
      # Get operator code info
      opcode_obj = model.operator_codes[fb_op.opcodeIndex]

      # Resolve tensor indices to Tensor objects
      inputs = [sg.tensors[i]
                for i in fb_op.inputs] if fb_op.inputs is not None else []
      outputs = [sg.tensors[i]
                 for i in fb_op.outputs] if fb_op.outputs is not None else []

      # Create Operator wrapping the OperatorT; all fields preserved in _fb
      op = Operator(
          _fb=fb_op,
          opcode=opcode_obj.builtin_code,
          inputs=inputs,
          outputs=outputs,
          custom_code=opcode_obj.custom_code,
          opcode_index=fb_op.opcodeIndex,
      )
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
      tflite.TensorType.INT64: np.int64,
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
    """Compile model using backing ModelT, preserving all fields."""
    # Use the backing ModelT directly---this preserves all fields we don't
    # explicitly handle (version, signature_defs, etc.)
    root = self.model._fb

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
    # Build lookup from existing OperatorCodes (from read()) to reuse their _fb
    existing_opcodes = {
        (oc.builtin_code, oc.custom_code): oc
        for oc in self.model.operator_codes
    }

    for sg in self.model.subgraphs:
      for op in sg.operators:
        key = (op.opcode, op.custom_code)
        if key not in self._operator_codes:
          # Reuse existing OperatorCodeT if available (preserves deprecated_builtin_code)
          if key in existing_opcodes:
            self._operator_codes[key] = existing_opcodes[key]._fb
          else:
            # Create new OperatorCodeT for newly added operators
            opcode = tflite.OperatorCodeT()
            opcode.builtinCode = op.opcode
            if op.custom_code:
              opcode.customCode = op.custom_code
            self._operator_codes[key] = opcode

  def _compile_subgraph(self, sg: Subgraph) -> tflite.SubGraphT:
    """Compile subgraph using backing SubGraphT, preserving all fields."""
    # Use the backing SubGraphT directly---this preserves all fields we don't
    # explicitly handle (debug_metadata_index, etc.)
    sg_t = sg._fb

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
    """Compile operator using backing OperatorT, preserving all fields."""
    # Use the backing OperatorT directly---this preserves all fields we don't
    # explicitly handle (builtin_options, custom_options, intermediates, etc.)
    op_t = op._fb

    # Get opcode index
    key = (op.opcode, op.custom_code)
    opcode_index = list(self._operator_codes.keys()).index(key)
    op_t.opcodeIndex = opcode_index

    # Resolve tensor references to indices
    op_t.inputs = [tensor_to_index[id(inp)] for inp in op.inputs]
    op_t.outputs = [tensor_to_index[id(outp)] for outp in op.outputs]

    return op_t

  def _compile_tensor(self, tensor: Tensor) -> tflite.TensorT:
    """Compile tensor using backing TensorT, preserving all fields."""
    # Use the backing TensorT directly---this preserves all fields we don't
    # explicitly handle (is_variable, sparsity, shape_signature, has_rank, etc.)
    t = tensor._fb

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

    # Sync quantization: merge our Quantization object into _fb.quantization
    if tensor.quantization:
      if t.quantization is None:
        t.quantization = tflite.QuantizationParametersT()
      # Update only the fields we manage; other fields (min, max, details)
      # are preserved from the original _fb.quantization
      q = tensor.quantization
      scales = [q.scales] if isinstance(q.scales, (int, float)) else q.scales
      zeros = [q.zero_points] if isinstance(q.zero_points,
                                            int) else q.zero_points
      t.quantization.scale = scales
      t.quantization.zeroPoint = zeros
      if q.axis is not None:
        t.quantization.quantizedDimension = q.axis

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
