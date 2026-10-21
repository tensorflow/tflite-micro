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
"""Python package for TFLM Python Interpreter using standard C-API."""

import ctypes
import enum
import os
import sys
import numpy as np

from tflite_micro.tensorflow.lite.micro.python import (
    schema_py_generated as schema_fb,
)


def _load_tflm_library():
  """Finds and loads libtflm.so across google3, Bazel, and PyPI."""
  # 1. Try google3 runfiles:
  try:
    from google3.pyglib import resources  # pylint: disable=g-import-not-at-top

    for candidate in [
        "google3/third_party/tflite_micro/tensorflow/lite/micro/c_api/libtflm.so",
        "google3/third_party/tflite_micro/python/tflite_micro/libtflm.so",
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
        "tflite_micro/tensorflow/lite/micro/c_api/libtflm.so",
        "tflite_micro/python/tflite_micro/libtflm.so",
    ]:
      path = r.Rlocation(candidate)
      if path and os.path.exists(path):
        return ctypes.CDLL(path)
  except (ImportError, TypeError):
    pass

  # 3. Try directory adjacent to this file:
  adjacent_path = os.path.join(os.path.dirname(__file__), "libtflm.so")
  if os.path.exists(adjacent_path):
    return ctypes.CDLL(adjacent_path)

  # 4. Try c_api directory inside the installed tflite_micro wheel namespace:
  wheel_c_api_path = os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "tensorflow",
      "lite",
      "micro",
      "c_api",
      "libtflm.so",
  )
  if os.path.exists(wheel_c_api_path):
    return ctypes.CDLL(os.path.abspath(wheel_c_api_path))

  # 5. Try repo root relative path:
  rel_path = os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "tensorflow",
      "lite",
      "micro",
      "c_api",
      "libtflm.so",
  )
  if os.path.exists(rel_path):
    return ctypes.CDLL(os.path.abspath(rel_path))

  # 5. Fallback to system dynamic loader:
  return ctypes.CDLL("libtflm.so")


_lib = _load_tflm_library()

# Function signatures
_lib.TFLM_InterpreterCreate.restype = ctypes.c_void_p
_lib.TFLM_InterpreterCreate.argtypes = [
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
]

_lib.TFLM_InterpreterDestroy.restype = None
_lib.TFLM_InterpreterDestroy.argtypes = [ctypes.c_void_p]

_lib.TFLM_InterpreterInvoke.restype = ctypes.c_int
_lib.TFLM_InterpreterInvoke.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.TFLM_InterpreterReset.restype = ctypes.c_int
_lib.TFLM_InterpreterReset.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.TFLM_InterpreterPrintAllocations.restype = None
_lib.TFLM_InterpreterPrintAllocations.argtypes = [ctypes.c_void_p]

_lib.TFLM_InterpreterSetInputData.restype = ctypes.c_int
_lib.TFLM_InterpreterSetInputData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.TFLM_InterpreterGetOutputData.restype = ctypes.c_void_p
_lib.TFLM_InterpreterGetOutputData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.TFLM_InterpreterGetTensorData.restype = ctypes.c_void_p
_lib.TFLM_InterpreterGetTensorData.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.TFLM_InterpreterGetTensorDetails.restype = ctypes.c_int
_lib.TFLM_InterpreterGetTensorDetails.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.TFLM_InterpreterGetEvalTensorDetails.restype = ctypes.c_int
_lib.TFLM_InterpreterGetEvalTensorDetails.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]

# TfLiteType enum to NumPy dtype mapping
_TFLITE_TYPE_TO_NUMPY_DTYPE = {
    1: np.float32,
    2: np.int32,
    3: np.uint8,
    4: np.int64,
    6: np.bool_,
    7: np.int16,
    8: np.complex64,
    9: np.int8,
    10: np.float16,
    11: np.float64,
    12: np.complex128,
    13: np.uint64,
    16: np.uint32,
    17: np.uint16,
    18: np.int8,  # Int4
    20: np.int8,  # Int2
    21: np.uint8,  # UInt4
}

_NUMPY_DTYPE_TO_TFLITE_TYPE = {
    np.dtype(np.float32): 1,
    np.dtype(np.int32): 2,
    np.dtype(np.uint8): 3,
    np.dtype(np.int64): 4,
    np.dtype(np.bool_): 6,
    np.dtype(np.int16): 7,
    np.dtype(np.complex64): 8,
    np.dtype(np.int8): 9,
    np.dtype(np.float16): 10,
    np.dtype(np.float64): 11,
    np.dtype(np.complex128): 12,
    np.dtype(np.uint64): 13,
    np.dtype(np.uint32): 16,
    np.dtype(np.uint16): 17,
}


def convert_bytearray_to_object(model_bytearray):
  """Converts a tflite model from a bytearray to an object for parsing."""
  model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
  return schema_fb.ModelT.InitFromObj(model_object)


def get_builtin_code_from_operator_code(opcode):
  """Return the builtin code of the given operator code."""
  if hasattr(opcode, 'BuiltinCode') and callable(opcode.BuiltinCode):
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
    err_buf = ctypes.create_string_buffer(512)
    status = _lib.TFLM_InterpreterInvoke(self._handle, err_buf, len(err_buf))
    if status != 0:
      raise RuntimeError(err_buf.value.decode("utf-8") or "Interpreter invocation failed.")
    return status

  def reset(self):
    """Reset the model state."""
    err_buf = ctypes.create_string_buffer(512)
    status = _lib.TFLM_InterpreterReset(self._handle, err_buf, len(err_buf))
    if status != 0:
      raise RuntimeError(err_buf.value.decode("utf-8") or "Failed to reset interpreter.")
    return status

  def set_input(self, input_data, index):
    """Set input data into input tensor."""
    if input_data is None:
      raise ValueError("Input data must not be None")
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    # Convert input_data to NumPy array if it isn't already
    if not isinstance(input_data, np.ndarray):
      input_data = np.array(input_data)

    if not input_data.flags["C_CONTIGUOUS"]:
      input_data = np.ascontiguousarray(input_data)

    type_code = _NUMPY_DTYPE_TO_TFLITE_TYPE.get(input_data.dtype, -1)

    dims = (ctypes.c_int * input_data.ndim)(*input_data.shape)
    err_buf = ctypes.create_string_buffer(512)
    ptr = input_data.ctypes.data_as(ctypes.c_void_p)

    status = _lib.TFLM_InterpreterSetInputData(
        self._handle,
        index,
        ptr,
        input_data.nbytes,
        type_code,
        dims,
        input_data.ndim,
        err_buf,
        len(err_buf),
    )
    if status != 0:
      err_msg = err_buf.value.decode("utf-8")
      if "Cannot set tensor" in err_msg or "numpy array had" in err_msg:
        raise ValueError(err_msg)
      if "out of bound" in err_msg:
        raise IndexError(err_msg)
      raise RuntimeError(err_msg)

  def get_output(self, index):
    """Get data from output tensor."""
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    details = self.get_output_details(index)
    byte_size = ctypes.c_size_t()
    err_buf = ctypes.create_string_buffer(512)
    ptr = _lib.TFLM_InterpreterGetOutputData(
        self._handle, index, ctypes.byref(byte_size), err_buf, len(err_buf)
    )
    if not ptr:
      raise IndexError(err_buf.value.decode("utf-8") or "Output tensor index out of bounds.")

    raw_buf = (ctypes.c_char * byte_size.value).from_address(ptr)
    arr = np.frombuffer(raw_buf, dtype=details["dtype"]).reshape(details["shape"])
    return arr.copy()

  def GetTensor(self, tensor_index, subgraph_index=0):
    """Retrieve intermediate tensor data when using kPreserveAllTensors."""
    details = self._get_eval_tensor_details(tensor_index, subgraph_index)
    byte_size = ctypes.c_size_t()
    err_buf = ctypes.create_string_buffer(512)
    ptr = _lib.TFLM_InterpreterGetTensorData(
        self._handle,
        tensor_index,
        subgraph_index,
        ctypes.byref(byte_size),
        err_buf,
        len(err_buf),
    )
    if not ptr:
      raise RuntimeError(err_buf.value.decode("utf-8") or "Failed to get tensor data.")

    raw_buf = (ctypes.c_char * byte_size.value).from_address(ptr)
    arr = np.frombuffer(raw_buf, dtype=details["dtype"]).reshape(details["shape"])
    details["tensor_data"] = arr.copy()
    return details

  def _get_eval_tensor_details(self, tensor_index, subgraph_index):
    out_type = ctypes.c_int()
    max_dims = 16
    out_dims = (ctypes.c_int * max_dims)()
    out_num_dims = ctypes.c_size_t()
    err_buf = ctypes.create_string_buffer(512)

    status = _lib.TFLM_InterpreterGetEvalTensorDetails(
        self._handle,
        tensor_index,
        subgraph_index,
        ctypes.byref(out_type),
        out_dims,
        ctypes.byref(out_num_dims),
        max_dims,
        err_buf,
        len(err_buf),
    )
    if status != 0:
      raise RuntimeError(err_buf.value.decode("utf-8") or "Failed to get eval tensor details.")

    shape = np.array([out_dims[i] for i in range(out_num_dims.value)], dtype=np.int32)
    dtype = _TFLITE_TYPE_TO_NUMPY_DTYPE.get(out_type.value)
    return {"dtype": dtype, "shape": shape}

  def _get_tensor_details(self, is_input, index):
    out_type = ctypes.c_int()
    max_dims = 16
    out_dims = (ctypes.c_int * max_dims)()
    out_num_dims = ctypes.c_size_t()

    max_quant = 512
    out_scales = (ctypes.c_float * max_quant)()
    out_zero_points = (ctypes.c_int32 * max_quant)()
    out_num_quant = ctypes.c_size_t()
    out_quant_dim = ctypes.c_int()
    err_buf = ctypes.create_string_buffer(512)

    status = _lib.TFLM_InterpreterGetTensorDetails(
        self._handle,
        1 if is_input else 0,
        index,
        ctypes.byref(out_type),
        out_dims,
        ctypes.byref(out_num_dims),
        max_dims,
        out_scales,
        out_zero_points,
        ctypes.byref(out_num_quant),
        max_quant,
        ctypes.byref(out_quant_dim),
        err_buf,
        len(err_buf),
    )
    if status != 0:
      err_msg = err_buf.value.decode("utf-8")
      if "out of bound" in err_msg:
        raise IndexError(err_msg)
      raise ValueError(err_msg)

    dtype = _TFLITE_TYPE_TO_NUMPY_DTYPE.get(out_type.value)
    shape = np.array([out_dims[i] for i in range(out_num_dims.value)], dtype=np.int32)
    scales = np.array([out_scales[i] for i in range(out_num_quant.value)], dtype=np.float32)
    zero_points = np.array([out_zero_points[i] for i in range(out_num_quant.value)], dtype=np.int32)

    return {
        "dtype": dtype,
        "shape": shape,
        "quantization_parameters": {
            "scales": scales,
            "zero_points": zero_points,
            "quantized_dimension": out_quant_dim.value,
        },
    }

  def get_input_details(self, index):
    """Get input tensor information."""
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")
    return self._get_tensor_details(is_input=True, index=index)

  def get_output_details(self, index):
    """Get output tensor information."""
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")
    return self._get_tensor_details(is_input=False, index=index)
