# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
""" Tensor class """

from typing import Dict, Optional
import string
import textwrap

from tflite_micro.codegen import utils
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb

_TENSOR_TYPES: Dict[int, str] = {
    schema_fb.TensorType.FLOAT16: "kTfLiteFloat16",
    schema_fb.TensorType.FLOAT32: "kTfLiteFloat32",
    schema_fb.TensorType.FLOAT64: "kTfLiteFloat64",
    schema_fb.TensorType.INT16: "kTfLiteInt16",
    schema_fb.TensorType.UINT16: "kTfLiteUInt16",
    schema_fb.TensorType.INT32: "kTfLiteInt32",
    schema_fb.TensorType.UINT32: "kTfLiteUInt32",
    schema_fb.TensorType.UINT8: "kTfLiteUInt8",
    schema_fb.TensorType.INT8: "kTfLiteInt8",
    schema_fb.TensorType.INT64: "kTfLiteInt64",
    schema_fb.TensorType.UINT64: "kTfLiteUInt64",
    schema_fb.TensorType.STRING: "kTfLiteString",
    schema_fb.TensorType.BOOL: "kTfLiteBool",
    schema_fb.TensorType.COMPLEX64: "kTfLiteComplex64",
    schema_fb.TensorType.COMPLEX128: "kTfLiteComplex128",
    schema_fb.TensorType.RESOURCE: "kTfLiteResource",
    schema_fb.TensorType.VARIANT: "kTfLiteVariant",
    schema_fb.TensorType.INT4: "kTfLiteInt4",
}


class Buffer(object):
  """ This buffer could be either a static array or a pointer into the arena """

  def __init__(self, buffer_name: str, buffer: schema_fb.BufferT):
    # TODO(rjascani): Get arena allocation offsets from preprocessor
    self._buffer_name = buffer_name
    self._buffer = buffer

  @property
  def address(self) -> str:
    if self._buffer is None or self._buffer.data is None:
      # TODO(rjascani): This needs to point into the arena
      return f"nullptr /* {self._buffer_name} */"
    return f"&{self._buffer_name}"

  def generate_c_buffer_array(self, indent: str) -> str:
    if self._buffer is None or self._buffer.data is None:
      return f"// {self._buffer_name} is located in the arena\n"

    buffer_template = string.Template(
        "alignas(16) uint8_t ${buffer_name}[${size}] = {\n"
        "${body}\n"
        "};\n")

    byte_strs = ['0x{:02X}'.format(b) for b in self._buffer.data]

    lines = []
    for byte_strs_for_line in utils.split_into_chunks(byte_strs, 12):
      bytes_segment = ', '.join(byte_strs_for_line)
      lines.append(f'    {bytes_segment},')

    return textwrap.indent(
        buffer_template.substitute(buffer_name=self._buffer_name,
                                   size=len(self._buffer.data),
                                   body='\n'.join(lines)), indent)


class Tensor(object):

  def __init__(self, buffer: Buffer, tensor: schema_fb.TensorT):
    self._buffer = buffer
    self._tensor: schema_fb.TensorT = tensor

  @property
  def buffer_index(self) -> bool:
    return self._tensor.buffer

  @property
  def buffer(self) -> Buffer:
    return self._buffer

  @property
  def has_shape(self) -> bool:
    return self._tensor.shape is not None

  @property
  def needs_zero_length_int_array(self) -> bool:
    return not self.has_shape

  def generate_c_tensor_dims(self, type_name: str, tensor_name: str) -> str:
    if not self.has_shape:
      return f"// No data dims necessary for {tensor_name}"
    return utils.IntArray(self._tensor.shape).generate_c_struct(
        type_name + "Dims", tensor_name + "_dims")

  def generate_c_tensor_init(self, tflite_tensor_name: str,
                             tensor_name: str) -> str:
    init_template = string.Template(
        "${tflite_tensor_name} = TfLiteEvalTensor{\n"
        "    .data = {.data = static_cast<void*>(${data})},\n"
        "    .dims = ${dims},\n"
        "    .type = ${tflite_type}};")
    dims = "reinterpret_cast<TfLiteIntArray*>(&{})".format(
        f"{tensor_name}_dims" if self._tensor.
        shape is not None else "zero_length_int_array")

    return init_template.substitute(
        tflite_tensor_name=tflite_tensor_name,
        tensor_name=tensor_name,
        data=self._buffer.address,
        dims=dims,
        tflite_type=_TENSOR_TYPES[self._tensor.type])
