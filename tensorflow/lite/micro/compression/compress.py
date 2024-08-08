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

"""Reduces the number of weights in a .tflite model using various strategies."""

# Usage information:
# Default:
#   `bazel run tensorflow/lite/micro/tools:compress -- \
#     --input_model_path=</path/to/my_model.tflite>` \
#     --output_model_path=</path/to/output.tflite>`


from tensorflow.lite.micro.compression import metadata_flatbuffer_py_generated as compression_schema
from tensorflow.lite.python import schema_py_generated as tflite_schema

from absl import app
from absl import flags
from absl import logging
import bitarray
import bitarray.util
import numpy as np
import flatbuffers
import sklearn.cluster
import struct


_INPUT_MODEL_PATH = flags.DEFINE_string(
    "input_model_path",
    None,
    ".tflite input model path",
    required=True,
)

_TEST_COMPRESSED_MODEL = flags.DEFINE_bool(
    "test_compressed_model",
    False,
    "optional config to test models with random data and"
    " report on the differences in output.",
)

_OUTPUT_MODEL_PATH = flags.DEFINE_string(
    "output_model_path",
    None,
    ".tflite output path. Leave blank if same as input+.compressed.tflite",
)


def read_model(path):
  with open(path, 'rb') as file:
    buffer = bytearray(file.read())
  return tflite_schema.ModelT.InitFromPackedBuf(buffer, 0)


def write_model(model, path):
  builder = flatbuffers.Builder(32)
  root = model.Pack(builder)
  builder.Finish(root)
  buffer: bytearray = builder.Output()

  with open(path, 'wb') as file:
    file.write(buffer)


def pack_compression_metadata(m):
  builder = flatbuffers.Builder(32)
  root = m.Pack(builder)
  builder.Finish(root)
  buffer: bytearray = builder.Output()
  return buffer


def pack_lut_indexes(indexes, bitwidth):
  """Pack the sequence of integers given in `indexes` into bitwidth-wide fields
  in a buffer, and return the buffer. Raise an OverflowError if any element
  does not fit into a bitwidth-wide field. """
  ba = bitarray.bitarray(endian="big")
  for i in indexes:
    field = bitarray.util.int2ba(i, length=bitwidth, endian="big")
    ba.extend(field)
  return ba.tobytes()


def pack_lut_values(values, struct_format):
  """Pack the `values` into a buffer of bytes, using a `struct_format`
  character from the standard module `struct` to determine the type of values
  and corresponding encoding into bytes. Always little-endian byte order.
  """
  buffer = bytearray()
  little_endian = "<"
  packer = struct.Struct(little_endian + struct_format)
  for v in values:
    buffer.extend(packer.pack(v))
  return buffer


def unpack_buffer_values(data, struct_format):
  little_endian = "<"
  unpacker = struct.Struct(little_endian + struct_format)
  values = [v[0] for v in unpacker.iter_unpack(bytes(data))]
  return values


def tensor_type_to_struct_format(type):
  m = {
    tflite_schema.TensorType.INT8: "b",
    tflite_schema.TensorType.INT16: "h",
    tflite_schema.TensorType.FLOAT32: "f",
  }
  return m[type]


def bq(sequence, num_values):
  """Quantize a sequence of integers, minimizing the total error using k-means
  clustering.

  Parameters:
    sequence :list - a sequence of integers to be quanized
    num_values :int - the number of quantization levels

  Returns:
    (indexes, values): a tuple with the list of indexes and list of values
  """
  sequence = np.array(sequence).reshape(-1, 1)
  kmeans = sklearn.cluster.KMeans(n_clusters=num_values,
                                  random_state=0).fit(sequence)
  values = kmeans.cluster_centers_.flatten()
  values = np.round(values).astype(int).tolist()
  indexes = kmeans.predict(sequence).tolist()
  return (indexes, values)


def compress_tensor(subgraph_id, tensor_id, model):
  subgraph = model.subgraphs[subgraph_id]
  tensor = subgraph.tensors[tensor_id]
  struct_format = tensor_type_to_struct_format(tensor.type)
  buffer_id = tensor.buffer
  buffer = model.buffers[buffer_id]
  sequence = unpack_buffer_values(buffer.data, struct_format)
  bitwidth = 2
  indexes, values = bq(sequence, 2 ** bitwidth)

  # append index buffer
  buffer = tflite_schema.BufferT()
  buffer.data = pack_lut_indexes(indexes, bitwidth)
  model.buffers.append(buffer)
  index_id = len(model.buffers) - 1

  # append value buffer
  buffer = tflite_schema.BufferT()
  buffer.data = pack_lut_values(values, struct_format)
  model.buffers.append(buffer)
  value_id = len(model.buffers) - 1

  # create metadata
  lut_tensor = compression_schema.LutTensorT()
  lut_tensor.subgraph = subgraph_id
  lut_tensor.tensor = tensor_id
  lut_tensor.indexBitwidth = bitwidth
  lut_tensor.indexBuffer = index_id
  lut_tensor.valueBuffer = value_id

  return lut_tensor


def compress_fully_connected(subgraph_id, operator_id, model):
  # On a fully_connected operator, we compress the 2nd
  subgraph = model.subgraphs[subgraph_id]
  operator = subgraph.operators[operator_id]
  tensor_id_2 = operator.inputs[1]
  # tensor_id_3 = operator.inputs[2]
  lut_tensor_2 = compress_tensor(subgraph_id, tensor_id_2, model)
  # lut_tensor_3 = compress_tensor(subgraph_id, tensor_id_2, model)
  return (lut_tensor_2,)


def get_opcode_compressions(model):
  """Return a map of operator_code indexes to compression functions, for those
  operators we wish to and know how to compress.
  """
  compressable = {tflite_schema.BuiltinOperator.FULLY_CONNECTED: compress_fully_connected}
  compressions = {}
  for index, code in enumerate(model.operatorCodes):
    if code.builtinCode in compressable:
      compressions[index] = compressable[code.builtinCode]
  return compressions


def compress(model):
  # Walk op codes, identify those we compress, note index
  # Walk operators, match op code indexes, note tensors to compress
  # Walk those tensors, creating LUTs in buffers and metadata

  compressions = get_opcode_compressions(model)

  lut_tensors = []

  for subgraph_id, subgraph in enumerate(model.subgraphs):
    for operator_id, operator in enumerate(subgraph.operators):
      fn = compressions.get(operator.opcodeIndex)
      if fn is not None:
        result = fn(subgraph_id, operator_id, model)
        if result is not None:
          lut_tensors.extend(result)

  compression_metadata = compression_schema.MetadataT()
  compression_metadata.lutTensors = lut_tensors

  return compression_metadata


def main(_) -> None:
  output_model_path = _OUTPUT_MODEL_PATH.value or (
      _INPUT_MODEL_PATH.value.split(".tflite")[0] + ".compressed.tflite")
  logging.info("compressing %s to %s", _INPUT_MODEL_PATH.value, output_model_path)

  model = read_model(_INPUT_MODEL_PATH.value)

  compression_metadata = compress(model)

  buffer = tflite_schema.BufferT()
  buffer.data = pack_compression_metadata(compression_metadata)
  model.buffers.append(buffer)

  metadata = tflite_schema.MetadataT()
  metadata.name = "COMPRESSION_METADATA"
  metadata.buffer = len(model.buffers) - 1
  model.metadata.append(metadata)

  write_model(model, output_model_path)


if __name__ == "__main__":
  app.run(main)
