# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Python package for TFLM Python Interpreter using internal _runtime.so."""

import ctypes
import enum
import os
import numpy as np

from tflite_micro.tensorflow.lite.micro.python import (
  schema_py_generated as schema_fb,
)


def _load_runtime_library():
  """Finds and loads _runtime.so across google3, Bazel, and PyPI."""
  # 1. Try google3 runfiles:
  try:
    from google3.pyglib import resources  # pylint: disable=g-import-not-at-top

    for candidate in [
      "google3/third_party/tflite_micro/python/tflite_micro/_runtime.so",
    ]:
      path = resources.GetResourceFilename(candidate)
      if os.path.exists(path):
        return ctypes.CDLL(path)
  except ImportError:
    pass

  # 2. Try Bazel runfiles:
  try:
    from rules_python.python.runfiles import runfiles  # pylint: disable=g-import-not-at-top

    r = runfiles.Create()
    for candidate in [
      "tflite_micro/python/tflite_micro/_runtime.so",
    ]:
      path = r.Rlocation(candidate)
      if path and os.path.exists(path):
        return ctypes.CDLL(path)
  except (ImportError, TypeError):
    pass

  # 3. Try directory adjacent to this file (PyPI wheel or local build):
  adjacent_path = os.path.join(os.path.dirname(__file__), "_runtime.so")
  if os.path.exists(adjacent_path):
    return ctypes.CDLL(adjacent_path)

  # 4. Fallback to system dynamic loader:
  return ctypes.CDLL("_runtime.so")


_lib = _load_runtime_library()

_TENSOR_INPUT = 0
_TENSOR_OUTPUT = 1
_TENSOR_EVAL = 2


class _TensorInfo(ctypes.Structure):
  _fields_ = [
    ("tensor_type", ctypes.c_int),
    ("data", ctypes.c_void_p),
    ("bytes", ctypes.c_size_t),
    ("dims", ctypes.POINTER(ctypes.c_int)),
    ("num_dims", ctypes.c_size_t),
    ("scales", ctypes.POINTER(ctypes.c_float)),
    ("zero_points", ctypes.POINTER(ctypes.c_int32)),
    ("num_quant_params", ctypes.c_size_t),
    ("quantized_dimension", ctypes.c_int),
  ]


def _bind(fn, argtypes, restype=ctypes.c_int):
  fn.argtypes = argtypes
  fn.restype = restype


_bind(
  _lib.TFLM_InterpreterCreate,
  [
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
  ],
  restype=ctypes.c_void_p,
)
_bind(_lib.TFLM_InterpreterDestroy, [ctypes.c_void_p], restype=None)
_bind(
  _lib.TFLM_InterpreterLastError, [ctypes.c_void_p], restype=ctypes.c_char_p
)
_bind(_lib.TFLM_InterpreterInvoke, [ctypes.c_void_p])
_bind(_lib.TFLM_InterpreterReset, [ctypes.c_void_p])
_bind(_lib.TFLM_InterpreterPrintAllocations, [ctypes.c_void_p], restype=None)
_bind(
  _lib.TFLM_InterpreterSetInputData,
  [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t,
  ],
)
_bind(
  _lib.TFLM_InterpreterGetTensor,
  [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(_TensorInfo),
  ],
)

# FlatBuffer TensorType enum to NumPy dtype mapping
_TENSOR_TYPE_TO_NUMPY_DTYPE = {
  schema_fb.TensorType.FLOAT32: np.float32,
  schema_fb.TensorType.FLOAT16: np.float16,
  schema_fb.TensorType.INT32: np.int32,
  schema_fb.TensorType.UINT8: np.uint8,
  schema_fb.TensorType.INT64: np.int64,
  schema_fb.TensorType.BOOL: np.bool_,
  schema_fb.TensorType.INT16: np.int16,
  schema_fb.TensorType.COMPLEX64: np.complex64,
  schema_fb.TensorType.INT8: np.int8,
  schema_fb.TensorType.FLOAT64: np.float64,
  schema_fb.TensorType.COMPLEX128: np.complex128,
  schema_fb.TensorType.UINT64: np.uint64,
  schema_fb.TensorType.UINT32: np.uint32,
  schema_fb.TensorType.UINT16: np.uint16,
  schema_fb.TensorType.INT4: np.int8,
  schema_fb.TensorType.INT2: np.int8,
  schema_fb.TensorType.UINT4: np.uint8,
}

_NUMPY_DTYPE_TO_TENSOR_TYPE = {
  np.dtype(dtype): tensor_type
  for tensor_type, dtype in _TENSOR_TYPE_TO_NUMPY_DTYPE.items()
  if tensor_type
  not in (
    schema_fb.TensorType.INT4,
    schema_fb.TensorType.INT2,
    schema_fb.TensorType.UINT4,
  )
}


def _copy_tensor_buffer(info, dtype, shape):
  if not info.data and info.bytes > 0:
    raise RuntimeError("Tensor buffer is null.")
  raw_buf = (ctypes.c_char * info.bytes).from_address(info.data or 0)
  return np.frombuffer(raw_buf, dtype=dtype).reshape(shape).copy()


def convert_bytearray_to_object(model_bytearray):
  """Converts a tflite model from a bytearray to an object for parsing."""
  model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
  return schema_fb.ModelT.InitFromObj(model_object)


def get_builtin_code_from_operator_code(opcode):
  """Return the builtin code of the given operator code."""
  if hasattr(opcode, "BuiltinCode") and callable(opcode.BuiltinCode):
    return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
  return max(opcode.builtinCode, opcode.deprecatedBuiltinCode)


def count_resource_variables(model):
  """Calculates the number of unique resource variables in a model."""
  if not isinstance(model, schema_fb.ModelT):
    model = convert_bytearray_to_object(model)
  unique_shared_names = set()
  for subgraph in model.subgraphs:
    if subgraph.operators is None:
      continue
    for op in subgraph.operators:
      builtin_code = get_builtin_code_from_operator_code(
        model.operatorCodes[op.opcodeIndex]
      )
      if builtin_code == schema_fb.BuiltinOperator.VAR_HANDLE:
        unique_shared_names.add(op.builtinOptions.sharedName)
  return len(unique_shared_names)


class InterpreterConfig(enum.Enum):
  """Interpreter configuration modes."""

  kAllocationRecording = 0
  kPreserveAllTensors = 1


class Interpreter(object):
  """TFLM Python Interpreter wrapping C shared library."""

  def __init__(
    self,
    model_data,
    custom_op_registerers,
    arena_size,
    intrepreter_config=InterpreterConfig.kAllocationRecording,
    alt_decompression_memory_size=0,
  ):
    if model_data is None:
      raise ValueError("Model must not be None")

    if not isinstance(custom_op_registerers, list) or not all(
      isinstance(s, str) for s in custom_op_registerers
    ):
      raise ValueError("Custom ops registerers must be a list of strings")

    if arena_size is None:
      arena_size = len(model_data) * 10

    num_resource_variables = count_resource_variables(model_data)

    # Convert custom_op_registerers to C char** array
    c_reg_array = None
    if custom_op_registerers:
      c_strings = [s.encode("utf-8") for s in custom_op_registerers]
      c_reg_array = (ctypes.c_char_p * len(c_strings))(*c_strings)

    err_buf = ctypes.create_string_buffer(512)
    # Retain model_data reference to prevent Python GC from releasing memory
    self._model_data = bytes(model_data)
    self._handle = _lib.TFLM_InterpreterCreate(
      self._model_data,
      len(self._model_data),
      c_reg_array,
      len(custom_op_registerers),
      arena_size,
      num_resource_variables,
      intrepreter_config.value,
      alt_decompression_memory_size,
      err_buf,
      len(err_buf),
    )

    if not self._handle:
      err_msg = err_buf.value.decode("utf-8") or "Unknown error"
      if "Python object to char *" in err_msg:
        raise ValueError(err_msg)
      raise RuntimeError(err_msg)

  def __del__(self):
    if hasattr(self, "_handle") and self._handle:
      _lib.TFLM_InterpreterDestroy(self._handle)
      self._handle = None

  def _check(self, status, error_cls=RuntimeError):
    if status == 0:
      return
    err_ptr = _lib.TFLM_InterpreterLastError(self._handle)
    err_msg = err_ptr.decode("utf-8") if err_ptr else "Unknown error"
    if "out of bound" in err_msg:
      raise IndexError(err_msg)
    if "Cannot set tensor" in err_msg or "numpy array had" in err_msg:
      raise ValueError(err_msg)
    raise error_cls(err_msg)

  @classmethod
  def from_file(
    cls,
    model_path,
    custom_op_registerers=None,
    arena_size=None,
    intrepreter_config=InterpreterConfig.kAllocationRecording,
    alt_decompression_memory_size=0,
  ):
    """Instantiates a TFLM interpreter from a model .tflite filepath."""
    if custom_op_registerers is None:
      custom_op_registerers = []

    if model_path is None or not os.path.isfile(model_path):
      raise ValueError("Invalid model file path")

    with open(model_path, "rb") as f:
      model_data = f.read()

    return cls(
      model_data,
      custom_op_registerers,
      arena_size,
      intrepreter_config,
      alt_decompression_memory_size,
    )

  @classmethod
  def from_bytes(
    cls,
    model_data,
    custom_op_registerers=None,
    arena_size=None,
    intrepreter_config=InterpreterConfig.kAllocationRecording,
    alt_decompression_memory_size=0,
  ):
    """Instantiates a TFLM interpreter from a model in byte array."""
    if custom_op_registerers is None:
      custom_op_registerers = []

    return cls(
      model_data,
      custom_op_registerers,
      arena_size,
      intrepreter_config,
      alt_decompression_memory_size,
    )

  def print_allocations(self):
    """Prints arena allocations."""
    _lib.TFLM_InterpreterPrintAllocations(self._handle)

  def invoke(self):
    """Invoke the TFLM interpreter to run an inference."""
    status = _lib.TFLM_InterpreterInvoke(self._handle)
    self._check(status)
    return status

  def reset(self):
    """Reset the model state."""
    status = _lib.TFLM_InterpreterReset(self._handle)
    self._check(status)
    return status

  def set_input(self, input_data, index):
    """Set input data into input tensor."""
    if input_data is None:
      raise ValueError("Input data must not be None")
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    if not isinstance(input_data, np.ndarray):
      input_data = np.array(input_data)

    if not input_data.flags["C_CONTIGUOUS"]:
      input_data = np.ascontiguousarray(input_data)

    tensor_type = _NUMPY_DTYPE_TO_TENSOR_TYPE.get(input_data.dtype, -1)
    dims = (ctypes.c_int * input_data.ndim)(*input_data.shape)
    ptr = input_data.ctypes.data_as(ctypes.c_void_p)

    status = _lib.TFLM_InterpreterSetInputData(
      self._handle,
      index,
      ptr,
      input_data.nbytes,
      tensor_type,
      dims,
      input_data.ndim,
    )
    self._check(status)

  def _get_tensor_info(self, kind, index, subgraph_index=0):
    info = _TensorInfo()
    status = _lib.TFLM_InterpreterGetTensor(
      self._handle, kind, index, subgraph_index, ctypes.byref(info)
    )
    default_cls = RuntimeError if kind == _TENSOR_EVAL else ValueError
    self._check(status, default_cls)
    dtype = _TENSOR_TYPE_TO_NUMPY_DTYPE.get(info.tensor_type)
    shape = np.array(
      [info.dims[i] for i in range(info.num_dims)], dtype=np.int32
    )
    return info, dtype, shape

  def get_output(self, index):
    """Get data from output tensor."""
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    info, dtype, shape = self._get_tensor_info(_TENSOR_OUTPUT, index)
    return _copy_tensor_buffer(info, dtype, shape)

  def GetTensor(self, tensor_index, subgraph_index=0):
    """Retrieve intermediate tensor data when using kPreserveAllTensors."""
    info, dtype, shape = self._get_tensor_info(
      _TENSOR_EVAL, tensor_index, subgraph_index
    )
    return {
      "dtype": dtype,
      "shape": shape,
      "tensor_data": _copy_tensor_buffer(info, dtype, shape),
    }

  def _get_tensor_details(self, kind, index):
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    info, dtype, shape = self._get_tensor_info(kind, index)
    scales = np.array(
      [info.scales[i] for i in range(info.num_quant_params)], dtype=np.float32
    )
    zero_points = np.array(
      [info.zero_points[i] for i in range(info.num_quant_params)],
      dtype=np.int32,
    )
    return {
      "dtype": dtype,
      "shape": shape,
      "quantization_parameters": {
        "scales": scales,
        "zero_points": zero_points,
        "quantized_dimension": info.quantized_dimension,
      },
    }

  def get_input_details(self, index):
    """Get input tensor information."""
    return self._get_tensor_details(_TENSOR_INPUT, index)

  def get_output_details(self, index):
    """Get output tensor information."""
    return self._get_tensor_details(_TENSOR_OUTPUT, index)
