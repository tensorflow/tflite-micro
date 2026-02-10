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
"""DECODE compression module."""

# Implements the DECODE operator compression scheme described in the
# "TFLM DECODE Operator Design" document, revised May 20, 2025.
#
# The DECODE operator transforms an encoded tensor, alongside a paired
# ancillary data tensor, into a tensor ready for use as input to any
# operator. For example, an encoded tensor might contain compressed
# data, while the paired ancillary data tensor holds the information
# necessary for decompression. The DECODE operator's output is a fully
# decompressed tensor.
#
# DECODE operators are inserted into the TfLite model subgraph
# immediately before each operation that uses a decodable tensor as
# input.
#
# Ancillary Data Tensor
#
# The ancillary data tensor contains the information necessary for
# decoding. It begins with a 16-byte DECODE Common Metadata (DCM)
# header, followed by decode-type-specific ancillary data.
#
# DECODE Common Metadata (DCM)
#
# Byte 0: Decode type
#   0-127:   TFLM-supported decode operations (see below)
#   128-255: Custom operations requiring application-registered
#            handlers
#
#   Supported decode types:
#
#   0: LUT decompression
#      All TFLM tensor types supported in reference and optimized
#      code.
#
#   1: Huffman decompression using Xtensa format decode tables
#      INT8 and INT16 tensor types only, in reference and optimized
#      code.
#
#   2: Pruning decompression
#      All TFLM tensor types supported in reference and optimized
#      code.
#
#   3-127: Reserved
#
#   128-255: Custom decode types
#      Requires user-supplied encoding module and decoding ancillary
#      data.
#
# Byte 1: DCM version (currently 1)
#
# Bytes 2-3: Reserved
#
# Bytes 4-15: User-defined
#   Used by TFLM decode types to avoid requiring additional alignment
#   of metadata or ancillary data.
#
# The 16-byte DCM size ensures that subsequent metadata and ancillary
# data are 128-bit aligned, which is required for some optimized
# decoding operations such as Xtensa LUT decompression.
#
# For TFLM decode types, ancillary data starts immediately after the
# DCM. For custom decode types, the location is determined by
# user-defined metadata.

from dataclasses import dataclass
from typing import Protocol


class DecodeType:
  """Decode operation type (0-255).

  Use predefined constants for built-in types or DecodeType.custom()
  for custom types:
      DecodeType.LUT        # 0
      DecodeType.HUFFMAN    # 1
      DecodeType.PRUNING    # 2
      DecodeType.custom(200)  # Custom type 128-255
  """

  # Built-in decode types (class variables set after class definition)
  LUT: 'DecodeType'
  HUFFMAN: 'DecodeType'
  PRUNING: 'DecodeType'

  def __init__(self, code: int, name: str = None):
    """Initialize DecodeType.

    Args:
        code: Integer code 0-255
        name: Optional name for the type. If not provided:
              - Codes 0-127: Named "TYPE_{code}"
              - Codes 128-255: Named "CUSTOM_{code}"
    """
    if not 0 <= code <= 255:
      raise ValueError(f"Decode type must be 0-255, got {code}")
    self.code = code

    # Auto-generate name if not provided
    if name is None:
      self.name = f"CUSTOM_{code}" if code >= 128 else f"TYPE_{code}"
    else:
      self.name = name

    self._is_custom = code >= 128

  @property
  def is_custom(self) -> bool:
    """True if this is a custom decode type (128-255)."""
    return self._is_custom

  @classmethod
  def custom(cls, code: int) -> 'DecodeType':
    """Create custom decode type (128-255).

    Args:
        code: Integer code 128-255

    Returns:
        DecodeType with name CUSTOM_{code}
    """
    if not 128 <= code <= 255:
      raise ValueError(f"Custom decode type must be 128-255, got {code}")
    return cls(code)

  def __int__(self):
    """Convert to integer for serialization."""
    return self.code

  def __eq__(self, other):
    if isinstance(other, DecodeType):
      return self.code == other.code
    return self.code == other

  def __repr__(self):
    return f"DecodeType.{self.name}({self.code})"


# Define built-in decode type constants
DecodeType.LUT = DecodeType(0, "LUT")
DecodeType.HUFFMAN = DecodeType(1, "HUFFMAN")
DecodeType.PRUNING = DecodeType(2, "PRUNING")


@dataclass
class DecodeCommonMetadata:
  """16-byte DECODE Common Metadata (DCM) header.

  Attributes:
    decode_type: Decode operation type. Use DecodeType constants or
                 DecodeType.custom(code) for custom types.
    version: DCM version (currently 1).
    user_data: 12 bytes of user-defined data (bytes 4-15 of DCM). Used by TFLM
               decode types to avoid requiring additional alignment of metadata
               or ancillary data.
  """
  decode_type: DecodeType
  version: int = 1
  user_data: bytes = b'\x00' * 12

  def to_bytes(self) -> bytes:
    """Serialize DCM to 16-byte sequence."""
    decode_code = int(self.decode_type)
    if not 0 <= self.version <= 255:
      raise ValueError(f"version must be 0-255, got {self.version}")
    if len(self.user_data) < 12:
      # Pad with zeros if user_data is too short
      user_data = self.user_data + b'\x00' * (12 - len(self.user_data))
    else:
      user_data = self.user_data[:12]

    result = bytearray(16)
    result[0] = decode_code
    result[1] = self.version
    # bytes 2-3 remain zero (reserved)
    result[4:16] = user_data
    return bytes(result)


class AncillaryDataSerializer(Protocol):
  """Protocol for objects that can serialize ancillary data."""

  def to_bytes(self) -> bytes:
    ...


@dataclass
class AncillaryDataTensor:
  """Complete Ancillary Data Tensor (ADT): DCM + decode-type-specific data.

  The ADT is stored as a buffer in the TFLite model. It begins with a 16-byte
  DCM header, followed by decode-type-specific ancillary data.

  Attributes:
    dcm: The DECODE Common Metadata header.
    ancillary_data: The decode-type-specific ancillary data, either as raw bytes
                    or as an object implementing the AncillaryDataSerializer
                    protocol. May be None if only the DCM is needed.
  """
  dcm: DecodeCommonMetadata
  ancillary_data: AncillaryDataSerializer | bytes | None = None

  def with_ancillary_data(
      self, data: AncillaryDataSerializer | bytes) -> 'AncillaryDataTensor':
    """Create new ADT with ancillary data added.

    Args:
      data: Ancillary data to add, either as raw bytes or as an object
            implementing AncillaryDataSerializer.

    Returns:
      New AncillaryDataTensor with the specified ancillary data.
    """
    return AncillaryDataTensor(self.dcm, data)

  def to_bytes(self) -> bytes:
    """Serialize entire ADT to bytes.

    Returns:
      Byte sequence containing DCM followed by ancillary data (if present).
    """
    dcm_bytes = self.dcm.to_bytes()
    if self.ancillary_data is None:
      return dcm_bytes
    if isinstance(self.ancillary_data, bytes):
      return dcm_bytes + self.ancillary_data
    return dcm_bytes + self.ancillary_data.to_bytes()
