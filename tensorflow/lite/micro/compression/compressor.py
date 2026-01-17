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
"""Compression plugin interface."""

from dataclasses import dataclass
from typing import Protocol

from tflite_micro.tensorflow.lite.micro.compression import decode
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec


class CompressionError(Exception):
  """Raised when compression fails for the reason documented in the message."""

  def __init__(self, message, wrapped_exception=None):
    if wrapped_exception:
      super().__init__(f"{message}: {str(wrapped_exception)}")
    else:
      super().__init__(message)
    self.original_exception = wrapped_exception


@dataclass
class CompressionResult:
  """Result of compressing a tensor.

  Attributes:
    encoded_data: The compressed tensor data (e.g., packed indices for LUT).
    ancillary_data: The complete ancillary data tensor bytes (DCM + type-specific
                    data). This is the full buffer contents for the ancillary
                    tensor.
  """
  encoded_data: bytes
  ancillary_data: bytes


class Compressor(Protocol):
  """Protocol that compression plugins must implement.

  Each compression method (LUT, Huffman, Pruning) provides a class implementing
  this protocol. The compress() function uses duck typing to call the plugin.
  """

  @property
  def decode_type(self) -> decode.DecodeType:
    """The DecodeType constant for this compression method."""
    ...

  def compress(
      self,
      tensor: model_editor.Tensor,
      method: spec.CompressionMethod,
  ) -> CompressionResult:
    """Compress a tensor according to the specified method.

    Args:
      tensor: The tensor to compress. Must have data (tensor.array is not None)
              and quantization parameters for axis inference.
      method: The compression method spec (e.g., LookUpTableCompression).

    Returns:
      CompressionResult with encoded tensor data and ancillary data bytes.

    Raises:
      CompressionError: If compression fails (e.g., too many unique values
                        for specified bitwidth, missing quantization, etc.).
    """
    ...
