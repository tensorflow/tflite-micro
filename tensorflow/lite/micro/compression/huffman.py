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
"""Huffman compression plugin (stub).

This module provides a placeholder for Huffman compression using Xtensa-format
decode tables. The actual implementation is not yet available.

Supported tensor types (when implemented): INT8, INT16
"""

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import decode
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec


class HuffmanCompressor:
  """Huffman compression plugin (stub).

  This stub exists to validate the plugin architecture. The actual Huffman
  compression algorithm using Xtensa-format decode tables is not yet
  implemented.
  """

  @property
  def decode_type(self) -> decode.DecodeType:
    """Returns DecodeType.HUFFMAN."""
    return decode.DecodeType.HUFFMAN

  def compress(
      self,
      tensor: model_editor.Tensor,
      method: spec.CompressionMethod,
  ) -> compressor.CompressionResult:
    """Compress a tensor using Huffman encoding.

    Args:
      tensor: The tensor to compress.
      method: Must be a HuffmanCompression instance.

    Returns:
      CompressionResult (not implemented).

    Raises:
      CompressionError: Always, since this is a stub.
    """
    raise compressor.CompressionError(
        "Huffman compression not yet implemented. "
        "This stub exists to validate the plugin architecture.")
