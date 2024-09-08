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
"""
Usage:
  bazel run tensorflow/lite/micro/tools:compress -- \\
      $(realpath <input.tflite>) [<output.tflite>]

Transform applicable tensors into compressed, look-up-table tensors. This is
the last stage of model compression. A prior stage must rewrite the elements of
those tensors to a small number of discrete values. This stage reduces such
tensors elements into indices into a value table.

Identify tensors to compress according to the criteria:
  1. command line argument: --tensors [0:]3,[0:]4
  2. metadata["COMPRESSION_INSTRUCTIONS"] json
  3. all inputs to operators known to understand compression (default)
"""

from dataclasses import dataclass
from functools import reduce
from typing import Sequence
import math

from tflite_micro.tensorflow.lite.micro.compression import (
    lib,
    model_facade,
    metadata_py_generated as schema,
)

import absl.app
import absl.flags
import bitarray
import bitarray.util
import flatbuffers


class MetadataBuilder:

  def __init__(self):
    self._metadata = schema.MetadataT()
    self._metadata.subgraphs = []

  def pack(self) -> bytearray:
    builder = flatbuffers.Builder(1 * 2**10)
    root = self._metadata.Pack(builder)
    builder.Finish(root)
    return builder.Output()

  def subgraph(self, index):
    """Return subgraph at index, adding subgraphs if necessary."""
    try:
      subgraph = self._metadata.subgraphs[index]
    except IndexError:
      need = index + 1 - len(self._metadata.subgraphs)
      for _ in range(0, need):
        subgraph = self._add_subgraph()
    return subgraph

  def add_lut_tensor(self, subgraph_id):
    """Add LUT tensor to the given subgraph and return it."""
    tensor = schema.LutTensorT()
    self.subgraph(subgraph_id).lutTensors.append(tensor)
    return tensor

  def _add_subgraph(self):
    subgraph = schema.SubgraphT()
    subgraph.lutTensors = []
    self._metadata.subgraphs.append(subgraph)
    return subgraph


def pack(indices: Sequence[int], bitwidth: int) -> bytes:
  """Pack an iterable of indices into a bytearray using bitwidth-sized fields.
  """
  endianness = "big"
  bits = bitarray.bitarray(endian=endianness)
  for i in indices:
    bits.extend(bitarray.util.int2ba(i, length=bitwidth, endian=endianness))
  return bits.tobytes()


def lut_compress(tensor: model_facade.Tensor, metadata: MetadataBuilder, *,
                 alt_axis: bool):
  """ Transform the given tensor into a compressed LUT tensor.
  """
  assert len(tensor.values) == reduce(lambda x, y: x * y, tensor.shape)

  # Identify levels per channel
  nr_channels = tensor.channel_count
  levels = []
  stride = len(tensor.values) // nr_channels
  for channel in range(0, nr_channels):
    if alt_axis:
      channel_values = tensor.values[channel::nr_channels]
    else:
      start = channel * stride
      end = start + stride
      channel_values = tensor.values[start:end]
    channel_levels = sorted(set(channel_values))
    levels.append(channel_levels)

  nr_levels = max((len(ch) for ch in levels))
  index_bitwidth = math.ceil(math.log2(nr_levels)) if nr_levels > 1 else 1

  # create and write value buffer with levels
  value_buffer = tensor.subgraph.model.add_buffer()
  for channel in range(0, nr_channels):
    values = levels[channel]
    values.extend([0] * (nr_levels - len(values)))
    value_buffer.extend_values(values, tensor.type)

  # rewrite original buffer with indices
  indices = []
  for i, value in enumerate(tensor.values):
    if alt_axis:
      channel = i % nr_channels
    else:
      channel = i // stride
    indices.append(levels[channel].index(value))
  tensor.buffer.data = pack(indices, index_bitwidth)

  # write metadata
  lut_tensor = metadata.add_lut_tensor(subgraph_id=tensor.subgraph.index)
  lut_tensor.tensor = tensor.index
  lut_tensor.valueBuffer = value_buffer.index
  lut_tensor.indexBitwidth = index_bitwidth


@dataclass
class TensorSpec:
  subgraph_id: int
  tensor_id: int


def strategy_lut_listed_tensors(tensors: Sequence[TensorSpec],
                                alt_axis_tensors: Sequence[TensorSpec]):
  """Return a strategy that lut-compresses each tensor listed in args.
  """

  def _strategy(model: model_facade.Model, metadata: MetadataBuilder):
    for spec in tensors:
      tensor = model.subgraphs[spec.subgraph_id].tensors[spec.tensor_id]
      lut_compress(tensor, metadata, alt_axis=False)
    for spec in alt_axis_tensors:
      tensor = model.subgraphs[spec.subgraph_id].tensors[spec.tensor_id]
      lut_compress(tensor, metadata, alt_axis=True)

  return _strategy


def compress_model(buffer, strategy):
  model = model_facade.read(buffer)
  metadata = MetadataBuilder()
  strategy(model, metadata)
  model.add_metadata(lib.METADATA_KEY, metadata.pack())
  return model.pack()


def compress_file(input_path, output_path, strategy):
  with open(input_path, "rb") as file:
    buffer = bytes(file.read())
  compressed = compress_model(buffer, strategy)
  with open(output_path, "wb") as file:
    file.write(compressed)


FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("tensors", None,
                         "List of [subgraph]tensor,... indices to compress")
absl.flags.DEFINE_string("alt_axis_tensors", None,
                         "List of [subgraph]tensor,... indices to compress")


def parse_tensors_flag(arg):
  if arg is None:
    return []

  specs = []
  for element in arg.split(","):
    parts = [int(part) for part in element.split(":")]
    if len(parts) == 1:
      specs.append(TensorSpec(subgraph_id=0, tensor_id=parts[0]))
    elif len(parts) == 2:
      specs.append(TensorSpec(subgraph_id=parts[0], tensor_id=parts[1]))

  return specs


def main(argv):
  try:
    input_path = argv[1]
  except IndexError:
    absl.app.usage()
    return 1

  try:
    output_path = argv[2]
  except IndexError:
    output_path = input_path.split(".tflite")[0] + ".compressed.tflite"

  specs = parse_tensors_flag(FLAGS.tensors)
  alt_axis_specs = parse_tensors_flag(FLAGS.alt_axis_tensors)
  strategy = strategy_lut_listed_tensors(specs, alt_axis_specs)

  print(f"compressing {input_path} to {output_path}")
  compress_file(input_path, output_path, strategy)

  return 0


if __name__ == "__main__":
  absl.app.run(main)
