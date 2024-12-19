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
"""Model compression library and CLI.

See USAGE.
"""

import bitarray
import bitarray.util
from dataclasses import dataclass, field
import sys
from typing import ByteString, Iterable, Optional

import absl.app
import absl.flags
import flatbuffers
import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import model_facade
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import metadata_py_generated as schema

USAGE = f"""\
Usage: compress.py --input <in.tflite> --spec <spec.yaml> [--output <out.tflite>]

Produce a compressed model from the input model by compressing tensors
according to the instructions in the spec file. The spec file lists the tensors
to compress, the compression methods to use on each tensor, and any parameters
for each compression method.

The spec file is a YAML-format file with a dictionary at the root, containing a
key "tensors" with a list of tensors to compress as its value. E.g.:

---
{spec.EXAMPLE_YAML_SPEC}
---

The only compression method currently implemented is "lut", i.e.,
Look-Up-Table. This method requires the tensor in the input model to have a
small number of unique values, fewer than or equal to 2**index_bitwidth. LUT
compression collects these values into a lookup table, and rewrites the tensor
as bitwidth-wide integer indices into that lookup table. Presumably, the input
model has been trained or preprocessed in a way that the tensor values
are binned into a meaningful, limited set.
"""

# A compressed model augments the usual .tflite flatbuffer with a flatbuffer of
# its own containing compression metadata, stored at the buffer index stored at
# the following key in the .tflite flatbuffer's metadata map.
TFLITE_METADATA_KEY = "COMPRESSION_METADATA"


class CompressionError(Exception):
  """Raised when compression fails for the reason documented in the message."""

  def __init__(self, message, wrapped_exception=None):
    super().__init__(f"{message}: {str(wrapped_exception)}")
    self.original_exception = wrapped_exception


class _MetadataBuilder:
  """Builder for the compression metadata flatbuffer."""

  def __init__(self):
    self._metadata = schema.MetadataT()
    self._metadata.subgraphs = []

  def compile(self) -> bytearray:
    """Packs the metadata into a binary array and returns it.
    """
    builder = flatbuffers.Builder(1 * 2**10)
    root = self._metadata.Pack(builder)
    builder.Finish(root)
    return builder.Output()

  def subgraph(self, index: int):
    """Return subgraph at index, adding subgraphs if necessary.
    """
    while len(self._metadata.subgraphs) <= index:
      self._add_subgraph()
    return self._metadata.subgraphs[index]

  def add_lut_tensor(self, subgraph_id: int):
    """Add LUT tensor to the given subgraph and return it.
    """
    tensor = schema.LutTensorT()
    self.subgraph(subgraph_id).lutTensors.append(tensor)
    return tensor

  def _add_subgraph(self):
    subgraph = schema.SubgraphT()
    subgraph.lutTensors = []
    self._metadata.subgraphs.append(subgraph)
    return subgraph


@dataclass
class _LutCompressedArray:
  compression_axis: Optional[int] = None
  lookup_tables: list[np.ndarray] = field(default_factory=list)
  indices: np.ndarray = field(default_factory=lambda: np.array([]))

  @property
  def index_bitwidth(self) -> int:
    """Returns the number of bits required to encode the indices."""
    if self.indices is None:
      raise ValueError

    max_index = int(np.max(self.indices))
    return max_index.bit_length() or 1


def _lut_compress_array(tensor: np.ndarray,
                        axis: Optional[int]) -> _LutCompressedArray:
  """Compresses the given tensor using lookup tables.

  Args:
      tensor (np.ndarray): The tensor to be compressed.

      axis (Optional[int]): The axis along which to compress the tensor. If an
          axis is given, a lookup table is created for each slice along the
          axis. If axis is None, a single lookup table is used for the entire
          tensor.

          Compressing a tensor with a lookup table per slice along a
          particular axis is analogous to quantizing a tensor with different
          quantization parameters per slice along a particular axis (dimension).

  Returns:
      _LutCompressedArray: An object containing the compressed tensor data,
      including the lookup tables and indices.
  """
  compressed = _LutCompressedArray()
  compressed.compression_axis = axis

  if axis is None:
    # Compute unique values and indices for the entire tensor
    values, indices = np.unique(tensor, return_inverse=True)
    compressed.lookup_tables.append(values)
    compressed.indices = indices.reshape(tensor.shape)
  else:
    # Iterate over slices along the compression axis
    slice_indices = []
    for slice in np.moveaxis(tensor, axis, 0):
      values, indices = np.unique(slice, return_inverse=True)
      compressed.lookup_tables.append(values)
      indices = indices.reshape(slice.shape)
      slice_indices.append(indices)

    # Reconstruct a tensor of indices from the slices
    stacked = np.stack(slice_indices, axis=0)
    compressed.indices = np.moveaxis(stacked, 0, axis)

  return compressed


def _check_lut_compression(compression) -> spec.LookUpTableCompression:
  if len(compression) != 1:
    raise CompressionError("Each tensor must have exactly one compression")
  if not isinstance(compression[0], spec.LookUpTableCompression):
    raise CompressionError('Only "lut" compression may be specified')

  return compression[0]


def _identify_compression_axis(tensor: model_facade._Tensor) -> Optional[int]:
  """Determines the axis along which to compress.

  The axis along which to compress is inferred from the tensor's quantization
  parameters.

  Returns:
    The axis along which to compress, or None to indicate one value table for
    the entire tensor.

  Raises:
    CompressionError: If the axis cannot be determined.
  """
  q = tensor.quantization
  if q is not None \
      and q.scale is not None \
      and q.quantizedDimension < len(tensor.shape):
    quantization_channels = len(q.scale)
    if quantization_channels == 1:
      # Use one value table for the entire tensor
      return None

    if quantization_channels == tensor.shape[q.quantizedDimension]:
      return q.quantizedDimension

  raise CompressionError(
      f"Invalid or no quanitzation parameters from which to "
      f"infer the axis along which tensor should be compressed.")


def _check_bitwidth(compressed: int, specified: int, spec: spec.Tensor):
  """Applies business logic regarding specified bitwidth.

  It is an error if the bitwidth required to compress a tensor exceeds the
  specified bitwith, and a warning if the tensor can be compressed in less than
  the specified bitwidth. The latter is allowed, and is not an error, to permit
  testing with larger bitwidths without re-binning a model.
  """
  if compressed > specified:
    raise CompressionError(
        f"index_bitwidth too small: {compressed} bits needed to "
        f"enumerate unique values in tensor specified in {spec}")
  elif compressed < specified:
    print(
        f"warning: index_bitwidth too large: only {compressed} "
        f"bits needed to enumerate unique values in tensor specified in {spec}",
        file=sys.stderr)


def _pack_indices(indices: np.ndarray, bitwidth: int) -> bytes:
  """Packs indices into a bytearray using bitwidth-sized fields.
  """
  endianness = "big"
  bits = bitarray.bitarray(endian=endianness)
  for i in indices.ravel():
    bits.extend(
        bitarray.util.int2ba(int(i), length=bitwidth, endian=endianness))
  return bits.tobytes()


def _pack_lookup_tables(tables: list[np.ndarray], table_len: int) -> bytearray:
  """Packs the value tables of a LutCompressedArray.

  Pack the value tables of a LutCompressedArray into a bytes object in the
  format writable to a value_table buffer in the .tflite flatbuffer. The
  tables are concatinated.
  """
  buffer = bytearray()
  for t in tables:
    padding_needed = table_len - len(t)
    padded = np.pad(t, (0, padding_needed), mode='constant', constant_values=0)
    buffer.extend(padded.tobytes())

  return buffer


def compress(model_in: ByteString, specs: Iterable[spec.Tensor]) -> bytearray:
  """Compresses a model .tflite flatbuffer.

  Args:
    model_in: the original, uncompressed .tflite flatbuffer
    specs: an iterable of compression specs, see module spec.py

  Returns:
    A compressed flatbuffer.
  """
  model = model_facade.read(model_in)
  metadata = _MetadataBuilder()

  for spec in specs:
    try:
      tensor = model.subgraphs[spec.subgraph].tensors[spec.tensor]
      lut_compression = _check_lut_compression(spec.compression)
      spec_bitwidth = lut_compression.index_bitwidth
      axis = _identify_compression_axis(tensor)
      compressed = _lut_compress_array(tensor.array, axis)
      _check_bitwidth(compressed.index_bitwidth, spec_bitwidth, spec)

      # overwrite tensor data with indices
      tensor.buffer.data = _pack_indices(compressed.indices, spec_bitwidth)

      # write value buffer
      value_buffer = model.add_buffer()
      value_buffer.data = _pack_lookup_tables(compressed.lookup_tables,
                                              2**spec_bitwidth)
      # add compression metadata for tensor
      lut_tensor = metadata.add_lut_tensor(subgraph_id=tensor.subgraph.index)
      lut_tensor.tensor = tensor.index
      lut_tensor.valueBuffer = value_buffer.index
      lut_tensor.indexBitwidth = spec_bitwidth

    except Exception as e:
      raise CompressionError(f"error compressing {spec}") from e

  # add compression metadata to model
  model.add_metadata(TFLITE_METADATA_KEY, metadata.compile())

  return model.compile()


def _fail_w_usage() -> int:
  absl.app.usage()
  return 1


FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("input", None, help="uncompressed .tflite flatbuffer")
absl.flags.DEFINE_string("spec", None, help="specfile (see module spec.py)")
absl.flags.DEFINE_string("output", None, help="compressed .tflite flatbuffer")


def main(argv):
  if len(argv) > 1:
    # no positional arguments accepted
    return _fail_w_usage()

  in_path = FLAGS.input
  if in_path is None:
    return _fail_w_usage()
  else:
    with open(in_path, "rb") as in_file:
      in_model = in_file.read()

  spec_path = FLAGS.spec
  if spec_path is None:
    return _fail_w_usage()
  else:
    with open(spec_path, "r") as spec_file:
      specs = spec.parse_yaml(spec_file.read())

  out_path = FLAGS.output
  if out_path is None:
    out_path = in_path.split(".tflite")[0] + ".compressed.tflite"

  compressed = compress(in_model, specs)

  with open(out_path, "wb") as out_file:
    out_file.write(compressed)

  return 0


if __name__ == "__main__":
  sys.modules['__main__'].__doc__ = USAGE  # for absl's use
  absl.app.run(main)
