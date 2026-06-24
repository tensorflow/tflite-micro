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
"""LUT (Look-Up Table) compression plugin."""

import sys
from dataclasses import dataclass, field
from typing import Optional

import bitarray
import bitarray.util
import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import decode
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec


@dataclass
class LutCompressedArray:
  """Intermediate representation of LUT-compressed data.

  Attributes:
    compression_axis: The axis along which compression was performed, or None
                      for per-tensor compression.
    lookup_tables: List of value lookup tables. One table for per-tensor
                   compression, or one per channel for per-channel compression.
    indices: Array of indices into the lookup tables, same shape as original.
  """
  compression_axis: Optional[int] = None
  lookup_tables: list[np.ndarray] = field(default_factory=list)
  indices: np.ndarray = field(default_factory=lambda: np.array([]))

  @property
  def index_bitwidth(self) -> int:
    """Returns the number of bits required to encode the indices."""
    if self.indices is None or self.indices.size == 0:
      raise ValueError("No indices to compute bitwidth from")
    max_index = int(np.max(self.indices))
    return max_index.bit_length() or 1


@dataclass
class LutAncillaryData:
  """LUT-specific ancillary data matching C++ decode_state_lut.cc format.

  The LUT ancillary data uses the DCM user_data bytes (4-15) plus value tables:
    - Byte 4: LUT version (currently 1)
    - Byte 5: Params (lower 3 bits = bitwidth, 1-7)
    - Byte 6: Value table channel stride (elements per channel)
    - Bytes 7-15: Reserved (zeros)
    - Bytes 16+: Value tables (concatenated, stride elements per channel)

  Attributes:
    lut_version: LUT format version (currently 1).
    bitwidth: Number of bits per index (1-7).
    value_table_stride: Number of elements per channel in value tables.
    value_tables: Packed value table data following the DCM.
  """
  lut_version: int = 1
  bitwidth: int = 4
  value_table_stride: int = 16
  value_tables: bytes = b''

  def __post_init__(self):
    if not 1 <= self.bitwidth <= 7:
      raise ValueError(f"bitwidth must be 1-7, got {self.bitwidth}")
    if not 0 <= self.value_table_stride <= 128:
      raise ValueError(
          f"value_table_stride must be 0-128, got {self.value_table_stride}")

  def to_user_data(self) -> bytes:
    """Serialize to 12-byte user_data for DCM bytes 4-15."""
    user_data = bytearray(12)
    user_data[0] = self.lut_version
    user_data[1] = self.bitwidth & 0x07
    user_data[2] = self.value_table_stride
    # Bytes 3-11 (DCM bytes 7-15) remain zero (reserved)
    return bytes(user_data)

  def to_bytes(self) -> bytes:
    """Serialize for use as AncillaryDataTensor.ancillary_data."""
    # This returns the type-specific data that follows the DCM header.
    # For LUT, that's just the value tables since user_data is in DCM.
    return self.value_tables


def compress_array(tensor: np.ndarray,
                   axis: Optional[int]) -> LutCompressedArray:
  """Compresses the given tensor using lookup tables.

  Args:
    tensor: The tensor to be compressed.
    axis: The axis along which to compress. If an axis is given, a lookup table
          is created for each slice along the axis. If axis is None, a single
          lookup table is used for the entire tensor.

          Compressing a tensor with a lookup table per slice along a particular
          axis is analogous to quantizing a tensor with different quantization
          parameters per slice along a particular axis (dimension).

  Returns:
    LutCompressedArray containing lookup tables and indices.
  """
  compressed = LutCompressedArray()
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


def identify_compression_axis(tensor: model_editor.Tensor) -> Optional[int]:
  """Determines the axis along which to compress.

  The axis along which to compress is inferred from the tensor's quantization
  parameters. Unquantized tensors use per-tensor compression.

  Args:
    tensor: The tensor to analyze.

  Returns:
    The axis along which to compress, or None to indicate one value table for
    the entire tensor.

  Raises:
    CompressionError: If the axis cannot be determined from quantization.
  """
  q = tensor.quantization
  if q is None:
    return None

  # model_editor wraps quantization, access scales/axis from wrapper
  scales = q.scales if isinstance(q.scales, list) else [q.scales]
  quantization_channels = len(scales)

  if quantization_channels == 1:
    return None

  if q.axis is not None and q.axis < len(tensor.shape):
    if quantization_channels == tensor.shape[q.axis]:
      return q.axis

  raise compressor.CompressionError(
      "Invalid or no quantization parameters from which to "
      "infer the axis along which tensor should be compressed.")


def check_bitwidth(compressed: int, specified: int, tensor_spec: spec.Tensor):
  """Validates that the specified bitwidth is sufficient.

  It is an error if the bitwidth required to compress a tensor exceeds the
  specified bitwith, and a warning if the tensor can be compressed in less than
  the specified bitwidth. The latter is allowed, and is not an error, to permit
  testing with larger bitwidths without re-binning a model.

  Args:
    compressed: The bitwidth required by the compressed data.
    specified: The bitwidth specified in the compression spec.
    tensor_spec: The tensor spec, for error messages.

  Raises:
    CompressionError: If specified bitwidth is too small.
  """
  if compressed > specified:
    raise compressor.CompressionError(
        f"index_bitwidth too small: {compressed} bits needed to "
        f"enumerate unique values in tensor specified in {tensor_spec}")
  elif compressed < specified:
    print(
        f"warning: index_bitwidth too large: only {compressed} "
        f"bits needed to enumerate unique values in tensor specified in "
        f"{tensor_spec}",
        file=sys.stderr)


def pack_indices(indices: np.ndarray, bitwidth: int) -> bytes:
  """Packs indices into a bytearray using bitwidth-sized fields.

  Args:
    indices: Array of indices to pack.
    bitwidth: Number of bits per index.

  Returns:
    Packed bytes with indices in big-endian bit order.
  """
  endianness = "big"
  bits = bitarray.bitarray(endian=endianness)
  for i in indices.ravel():
    bits.extend(
        bitarray.util.int2ba(int(i), length=bitwidth, endian=endianness))
  return bits.tobytes()


def pack_lookup_tables(tables: list[np.ndarray], table_len: int) -> bytes:
  """Packs the value tables of a LutCompressedArray.

  Pack the value tables of a LutCompressedArray into a bytes object in the
  format writable to a value_table buffer in the .tflite flatbuffer. The
  tables are concatenated.

  Args:
    tables: List of numpy arrays containing lookup table values.
    table_len: Length to pad each table to (typically 2**bitwidth).

  Returns:
    Packed bytes containing all tables concatenated.
  """
  buffer = bytearray()
  for t in tables:
    padding_needed = table_len - len(t)
    padded = np.pad(t, (0, padding_needed), mode='constant', constant_values=0)
    buffer.extend(padded.tobytes())
  return bytes(buffer)


class LutCompressor:
  """LUT compression plugin implementing the Compressor protocol."""

  @property
  def decode_type(self) -> decode.DecodeType:
    """Returns DecodeType.LUT."""
    return decode.DecodeType.LUT

  def compress(
      self,
      tensor: model_editor.Tensor,
      method: spec.CompressionMethod,
  ) -> compressor.CompressionResult:
    """Compress a tensor using LUT compression.

    Args:
      tensor: The tensor to compress.
      method: Must be a LookUpTableCompression instance.

    Returns:
      CompressionResult with packed indices and ancillary data.

    Raises:
      CompressionError: If compression fails.
    """
    if not isinstance(method, spec.LookUpTableCompression):
      raise compressor.CompressionError(
          f"LutCompressor requires LookUpTableCompression, got {type(method)}")

    if tensor.array is None:
      raise compressor.CompressionError("Tensor has no data to compress")

    spec_bitwidth = method.index_bitwidth
    axis = identify_compression_axis(tensor)
    compressed = compress_array(tensor.array, axis)
    # Note: check_bitwidth requires a spec.Tensor but we don't have it here.
    # We'll do a simpler check.
    actual_bitwidth = compressed.index_bitwidth
    if actual_bitwidth > spec_bitwidth:
      raise compressor.CompressionError(
          f"index_bitwidth too small: {actual_bitwidth} bits needed, "
          f"but only {spec_bitwidth} specified")
    elif actual_bitwidth < spec_bitwidth:
      print(
          f"warning: index_bitwidth larger than necessary: only "
          f"{actual_bitwidth} bits needed, but {spec_bitwidth} specified",
          file=sys.stderr)

    # Pack indices into bytes
    encoded_data = pack_indices(compressed.indices, spec_bitwidth)

    # Pack value tables
    table_len = max(len(t) for t in compressed.lookup_tables)
    value_tables_bytes = pack_lookup_tables(compressed.lookup_tables,
                                            table_len)

    # Build ancillary data
    lut_data = LutAncillaryData(
        lut_version=1,
        bitwidth=spec_bitwidth,
        value_table_stride=table_len,
        value_tables=value_tables_bytes,
    )

    # Build complete ancillary data tensor bytes: DCM header + value tables
    dcm = decode.DecodeCommonMetadata(
        decode_type=self.decode_type,
        user_data=lut_data.to_user_data(),
    )
    ancillary_data = dcm.to_bytes() + lut_data.to_bytes()

    return compressor.CompressionResult(
        encoded_data=encoded_data,
        ancillary_data=ancillary_data,
    )
