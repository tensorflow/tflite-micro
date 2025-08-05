# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

# This development tool prints compressed and uncompressed .tflite models to
# stdout in a human-readable, searchable, structured, text format. Helpful
# annotations (indexes of lists, names of operators, etc.) derived from the
# model are added as virtual fields with names beginning with an _underscore.
#
# Example usage:
#   bazel run //tensorflow/lite/micro/compression:view -- $(realpath model.tflite)
#
# Theory of operation:
# Convert the model into a Python dictionary, expressing the hierarchical nature
# of the model, and pretty print the dictionary. Please extend as needed for
# your use case.

from dataclasses import dataclass
from enum import Enum
import bitarray
import bitarray.util
import numpy as np
import os
import prettyprinter
import prettyprinter.doc
import sys
import textwrap

import absl.app

from tensorflow.lite.micro.compression import metadata_py_generated as compression_schema
from tensorflow.lite.python import schema_py_generated as tflite_schema

# Detect if running under Bazel by checking for BAZEL environment variables
is_bazel = 'BUILD_WORKING_DIRECTORY' in os.environ or 'BAZEL_TEST' in os.environ

if is_bazel:
  USAGE = textwrap.dedent("""\
    Usage: bazel run //tensorflow/lite/micro/compression:view -- <MODEL_PATH>
    
    Print a human-readable visualization of a .tflite model.
    Note: When running through Bazel, MODEL_PATH must be an absolute path.
    
    Example: bazel run //tensorflow/lite/micro/compression:view -- $(realpath model.tflite)"""
                          )
else:
  USAGE = textwrap.dedent(f"""\
    Usage: {os.path.basename(sys.argv[0])} <MODEL_PATH>
    
    Print a human-readable visualization of a .tflite model.""")


def print_model(model_path):
  with open(model_path, 'rb') as flatbuffer:
    d = create_dictionary(memoryview(flatbuffer.read()))
    prettyprinter.cpprint(d)


def main(argv):
  try:
    model_path = argv[1]
  except IndexError:
    sys.stderr.write(USAGE)
    sys.exit(1)

  print_model(model_path)


@dataclass
class MetadataReader:
  model: tflite_schema.ModelT
  buffer_index: int
  metadata: compression_schema.MetadataT

  @classmethod
  def build(cls, model: tflite_schema.ModelT):
    if model.metadata is None:
      return None

    for item in model.metadata:
      if _decode_name(item.name) == "COMPRESSION_METADATA":
        buffer_index = item.buffer
        buffer = model.buffers[buffer_index]
        metadata = compression_schema.MetadataT.InitFromPackedBuf(
            buffer.data, 0)
        if metadata.subgraphs is None:
          raise ValueError("Invalid compression metadata")
        return cls(model, buffer_index, metadata)
    else:
      return None

  def unpack(self):
    result = []
    for index, subgraph in enumerate(self.metadata.subgraphs):
      result.append({
          "_index": index,
          "lut_tensors": unpack_lut_metadata(subgraph.lutTensors),
      })
    return {"subgraphs": result}


def unpack_operators(model: tflite_schema.ModelT,
                     operators: list[tflite_schema.OperatorT]):
  result = []
  for index, op in enumerate(operators):
    opcode = model.operatorCodes[op.opcodeIndex]
    name = OPERATOR_NAMES[opcode.builtinCode]
    d = {
        "_operator": index,
        "opcode_index": op.opcodeIndex,
        "_opcode_name": name,
        "inputs": op.inputs,
        "outputs": op.outputs,
    }
    result.append(d)
  return result


def unpack_TensorType(type):
  attrs = [
      attr for attr in dir(tflite_schema.TensorType)
      if not attr.startswith("__")
  ]
  lut = {getattr(tflite_schema.TensorType, attr): attr for attr in attrs}
  return lut[type]


def _decode_name(name):
  """Returns name as a str or 'None'.

  The flatbuffer library returns names as bytes objects or None. This function
  returns a str, decoded from the bytes object, or None.
  """
  if name is None:
    return None
  else:
    return str(name, encoding="utf-8")


@dataclass
class TensorCoordinates:
  subgraph_ix: int
  tensor_index: int


class CompressionMethod(Enum):
  LUT = "LUT"


_NP_DTYPES = {
    tflite_schema.TensorType.FLOAT16: np.dtype("<f2"),
    tflite_schema.TensorType.FLOAT32: np.dtype("<f4"),
    tflite_schema.TensorType.FLOAT64: np.dtype("<f8"),
    tflite_schema.TensorType.INT8: np.dtype("<i1"),
    tflite_schema.TensorType.INT16: np.dtype("<i2"),
    tflite_schema.TensorType.INT32: np.dtype("<i4"),
    tflite_schema.TensorType.INT64: np.dtype("<i8"),
    tflite_schema.TensorType.UINT8: np.dtype("<u1"),
    tflite_schema.TensorType.UINT16: np.dtype("<u2"),
    tflite_schema.TensorType.UINT32: np.dtype("<u4"),
    tflite_schema.TensorType.UINT64: np.dtype("<u8"),
}

OPERATOR_NAMES = {
    code: name
    for name, code in tflite_schema.BuiltinOperator.__dict__.items()
}


class Codec:

  def __init__(self, reader: MetadataReader, model: tflite_schema.ModelT):
    self.reader = reader
    self.model = model

  def _tensor_metadata(self, tensor: TensorCoordinates):
    subgraph = self.reader.metadata.subgraphs[tensor.subgraph_ix]
    for metadata in subgraph.lutTensors:
      if tensor.tensor_index == metadata.tensor:
        return metadata
    else:
      return None

  def list_compressions(
      self, coordinates: TensorCoordinates) -> list[CompressionMethod]:
    metadata = self._tensor_metadata(coordinates)
    if metadata:
      return [CompressionMethod.LUT]
    else:
      return []

  def lookup_tables(self, coordinates: TensorCoordinates) -> np.ndarray:
    metadata = self._tensor_metadata(coordinates)
    if not metadata:
      return np.array([])

    model_subgraph = self.model.subgraphs[coordinates.subgraph_ix]
    model_tensor = model_subgraph.tensors[coordinates.tensor_index]
    value_buffer = self.model.buffers[metadata.valueBuffer]
    values = np.frombuffer(bytes(value_buffer.data),
                           dtype=_NP_DTYPES[model_tensor.type])
    values_per_table = 2**metadata.indexBitwidth
    tables = len(values) // values_per_table
    values = values.reshape((tables, values_per_table))

    return values


def unpack_tensors(tensors, subgraph_index: int, codec: Codec | None):
  result = []
  for index, t in enumerate(tensors):
    d = {
        "_tensor": index,
        "name": _decode_name(t.name),
        "type": unpack_TensorType(t.type),
        "shape": t.shape,
        "buffer": t.buffer,
    }

    if t.isVariable:
      d["is_variable"] = True
    else:
      # don't display this unusual field
      pass

    if t.quantization is not None and t.quantization.scale is not None:
      d["quantization"] = {
          "scale": t.quantization.scale,
          "zero": t.quantization.zeroPoint,
          "dimension": t.quantization.quantizedDimension,
      }
    result.append(d)

    if codec is not None:
      coordinates = TensorCoordinates(subgraph_ix=subgraph_index,
                                      tensor_index=index)
      d |= unpack_compression(coordinates, codec)

  return result


def unpack_compression(tensor: TensorCoordinates, codec: Codec) -> dict:
  result = {}

  compressions = codec.list_compressions(tensor)
  if compressions:
    result["_compressed"] = [c.name for c in compressions]
    metadata = codec._tensor_metadata(tensor)
    assert metadata is not None
    result["_value_buffer"] = metadata.valueBuffer
    result["_lookup_tables"] = codec.lookup_tables(tensor)

  return result


def unpack_subgraphs(model: tflite_schema.ModelT, codec: Codec | None):
  result = []
  for index, s in enumerate(model.subgraphs):
    d = {
        "_subgraph": index,
        "_operator_count": len(s.operators),
        "_tensor_count": len(s.tensors),
        "name": _decode_name(s.name),
        "operators": unpack_operators(model, s.operators),
        "tensors": unpack_tensors(s.tensors, subgraph_index=index,
                                  codec=codec),
    }
    result.append(d)
  return result


def unpack_opcodes(opcodes: list[tflite_schema.OperatorCodeT]) -> list:
  result = []
  for index, opcode in enumerate(opcodes):
    d: dict = {
        "_opcode_index": index,
        "_name": OPERATOR_NAMES[opcode.builtinCode],
        "builtin_code": opcode.builtinCode,
        "version": opcode.version,
    }
    if opcode.customCode is not None:
      d["custom_code"] = opcode.customCode
      del d["_name"]
    result.append(d)
  return result


def unpack_metadata(model: tflite_schema.ModelT):
  entries = []
  compression = MetadataReader.build(model)

  if model.metadata is None:
    return entries

  for m in model.metadata:
    d = {"name": _decode_name(m.name), "buffer": m.buffer}

    if compression and compression.buffer_index == m.buffer:
      d["_compression_metadata"] = compression.unpack()

    entries.append(d)

  return entries


def unpack_lut_metadata(lut_tensors):
  return [{
      "tensor": t.tensor,
      "value_buffer": t.valueBuffer,
      "index_bitwidth": t.indexBitwidth,
  } for t in sorted(lut_tensors, key=lambda x: x.tensor)]


def find_lut_info_for_buffer(buffer_index, model, compression_data):
  """Find LUT metadata for a given buffer index.
  
  Returns a dict with tensor_index, subgraph_index, and index_bitwidth if the
  buffer contains compressed indices, otherwise returns None.
  """
  if compression_data is None:
    return None

  for subgraph_idx, subgraph in enumerate(compression_data.metadata.subgraphs):
    for lut_tensor in subgraph.lutTensors:
      # Get the tensor to find which buffer contains the compressed indices
      tensor = model.subgraphs[subgraph_idx].tensors[lut_tensor.tensor]
      if tensor.buffer == buffer_index:
        return {
            "tensor_index": lut_tensor.tensor,
            "subgraph_index": subgraph_idx,
            "index_bitwidth": lut_tensor.indexBitwidth,
        }
  return None


def unpack_buffers(model, compression_data):
  buffers = []
  for index, buffer in enumerate(model.buffers):
    native = {
        "_buffer": index,
        "_bytes": len(buffer.data) if buffer.data is not None else 0,
    }

    if compression_data is not None and index == compression_data.buffer_index:
      native["_compression_metadata"] = True

    native["data"] = buffer.data

    # Check if this buffer contains compressed indices
    lut_info = find_lut_info_for_buffer(index, model, compression_data)
    if lut_info and buffer.data is not None:
      # Decode the indices from the buffer
      bstring = bitarray.bitarray()
      bstring.frombytes(bytes(buffer.data))
      bitwidth = lut_info["index_bitwidth"]
      chunks = [
          bstring[i:i + bitwidth]
          for i in range(0,
                         len(bstring) - bitwidth + 1, bitwidth)
      ]
      indices = [bitarray.util.ba2int(chunk) for chunk in chunks]

      # Convert indices to numpy array to match data field formatting
      indices_array = np.array(indices, dtype=np.uint8)

      native["_lut_indices"] = {
          "tensor": lut_info["tensor_index"],
          "bitwidth": bitwidth,
          "indices": indices_array,
      }

    buffers.append(native)

  return buffers


def create_dictionary(flatbuffer: memoryview) -> dict:
  """Returns a human-readable dictionary from the provided model flatbuffer.

  This function transforms a .tflite model flatbuffer into a Python dictionary.
  When pretty-printed, this dictionary offers an easily interpretable view of
  the model.
  """
  model = tflite_schema.ModelT.InitFromPackedBuf(flatbuffer, 0)
  compression_metadata = MetadataReader.build(model)
  codec = Codec(compression_metadata, model) if compression_metadata else None

  output = {
      "description": model.description,
      "version": model.version,
      "operator_codes": unpack_opcodes(model.operatorCodes),
      "metadata": unpack_metadata(model),
      "subgraphs": unpack_subgraphs(model, codec),
      "buffers": unpack_buffers(model, compression_metadata),
  }

  return output


@prettyprinter.register_pretty(np.ndarray)
def pretty_numpy_array(array, ctx):
  # Format array without ellipsis, similar to how buffer data is displayed
  string = np.array2string(array,
                           threshold=np.inf,
                           max_line_width=78,
                           separator=' ',
                           suppress_small=True)
  lines = string.splitlines()

  if len(lines) == 1:
    return lines[0]

  parts = list()
  parts.append(prettyprinter.doc.HARDLINE)
  for line in lines:
    parts.append(line)
    parts.append(prettyprinter.doc.HARDLINE)

  return prettyprinter.doc.nest(ctx.indent, prettyprinter.doc.concat(parts))


if __name__ == "__main__":
  sys.modules['__main__'].__doc__ = USAGE
  absl.app.run(main)
