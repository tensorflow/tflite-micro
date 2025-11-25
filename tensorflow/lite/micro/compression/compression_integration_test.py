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
"""Integration tests for compression with TFLM interpreter.

These tests verify that compressed models produce correct inference results
when run through the TFLM Python interpreter. Tests compress models and
compare outputs against uncompressed originals.

These tests only run when compression is enabled (--//:with_compression).
"""

import os
import unittest
import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import decode_insert
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


def _build_compressible_model(weight_shape=(4, 4)):
  """Build a model with clustered weights for compression testing.

  Args:
    weight_shape: Shape of the weight tensor as (rows, cols).

  Returns:
    A TFLite flatbuffer (bytes) containing a simple FULLY_CONNECTED model
    with weights that have only 4 unique values.
  """
  rows, cols = weight_shape

  # Create weights with only 4 unique values (compressible with 2-bit indices)
  pattern = np.array([1, 2, 3, 4], dtype=np.int8)
  weight_data = np.resize(pattern, (rows, cols))

  weights = model_editor.Tensor(
      shape=weight_shape,
      dtype=tflite.TensorType.INT8,
      data=weight_data,
      name="weights",
      quantization=model_editor.Quantization(scales=0.5, zero_points=0),
  )

  input_t = model_editor.Tensor(
      shape=(1, cols),
      dtype=tflite.TensorType.INT8,
      name="input",
  )
  output_t = model_editor.Tensor(
      shape=(1, rows),
      dtype=tflite.TensorType.INT8,
      name="output",
  )

  model = model_editor.Model(subgraphs=[
      model_editor.Subgraph(
          tensors=[weights],
          inputs=[input_t],
          outputs=[output_t],
          operators=[
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input_t, weights],
                  outputs=[output_t],
              )
          ],
      )
  ])
  return model.build()


class LutCompressionTest(tf.test.TestCase):
  """Integration tests for LUT (lookup table) compression."""

  def test_lut_compressed_model_matches_uncompressed(self):
    """LUT-compressed model produces same outputs as uncompressed."""
    flatbuffer = _build_compressible_model()

    # Create compression spec for weights tensor (index 0 in tensors list)
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.LookUpTableCompression(index_bitwidth=2)],
        )
    ]

    # Compress
    compressed_fb = compress.compress(flatbuffer, specs)

    # Run inference on both (convert bytearray to bytes for interpreter)
    uncompressed_interp = runtime.Interpreter.from_bytes(bytes(flatbuffer))
    compressed_interp = runtime.Interpreter.from_bytes(bytes(compressed_fb))

    # Test with multiple random inputs
    np.random.seed(42)
    for _ in range(10):
      test_input = np.random.randint(-128, 127, (1, 4), dtype=np.int8)

      uncompressed_interp.set_input(test_input, 0)
      uncompressed_interp.invoke()
      expected = uncompressed_interp.get_output(0)

      compressed_interp.set_input(test_input, 0)
      compressed_interp.invoke()
      actual = compressed_interp.get_output(0)

      self.assertAllEqual(expected, actual)

  def test_lut_decode_operators_present(self):
    """DECODE operators are inserted for LUT-compressed tensors."""
    flatbuffer = _build_compressible_model()

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.LookUpTableCompression(index_bitwidth=2)],
        )
    ]

    compressed_fb = compress.compress(flatbuffer, specs)
    model = model_editor.read(compressed_fb)
    sg = model.subgraphs[0]

    # Find DECODE operators
    decode_ops = [
        op for op in sg.operators if op.opcode == tflite.BuiltinOperator.CUSTOM
        and op.custom_code == decode_insert.DECODE_CUSTOM_OP_NAME
    ]

    self.assertGreater(len(decode_ops), 0,
                       "DECODE operators should be present")

  def test_lut_compressed_model_is_smaller(self):
    """LUT-compressed model is smaller than original.

    Uses a large enough weight tensor (64x64 = 4096 bytes) that compression
    savings outweigh the overhead from lookup tables and DECODE operators.
    With 2-bit indices, 4096 bytes becomes 1024 bytes of indices.
    """
    flatbuffer = _build_compressible_model(weight_shape=(64, 64))

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.LookUpTableCompression(index_bitwidth=2)],
        )
    ]

    compressed_fb = compress.compress(flatbuffer, specs)

    original_size = len(flatbuffer)
    compressed_size = len(compressed_fb)

    self.assertLess(
        compressed_size, original_size,
        f"Compressed model ({compressed_size} bytes) should be smaller than "
        f"original ({original_size} bytes)")


class HuffmanCompressionTest(tf.test.TestCase):
  """Integration tests for Huffman compression."""

  @unittest.skip("Huffman compression not yet implemented")
  def test_huffman_compressed_model_matches_uncompressed(self):
    """Huffman-compressed model produces same outputs as uncompressed."""
    pass

  @unittest.skip("Huffman compression not yet implemented")
  def test_huffman_decode_operators_present(self):
    """DECODE operators are inserted for Huffman-compressed tensors."""
    pass

  @unittest.skip("Huffman compression not yet implemented")
  def test_huffman_compressed_model_is_smaller(self):
    """Huffman-compressed model is smaller than original."""
    pass


class PruningCompressionTest(tf.test.TestCase):
  """Integration tests for pruning compression."""

  @unittest.skip("Pruning compression not yet implemented")
  def test_pruning_compressed_model_matches_uncompressed(self):
    """Pruning-compressed model produces same outputs as uncompressed."""
    pass

  @unittest.skip("Pruning compression not yet implemented")
  def test_pruning_decode_operators_present(self):
    """DECODE operators are inserted for pruning-compressed tensors."""
    pass

  @unittest.skip("Pruning compression not yet implemented")
  def test_pruning_compressed_model_is_smaller(self):
    """Pruning-compressed model is smaller than original."""
    pass


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  tf.test.main()
