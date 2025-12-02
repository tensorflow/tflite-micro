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


def _build_shared_weights_model():
  """Build a model where one compressed tensor is shared between two operators.

  Model structure:
    input1 -> [FC1 with weights1] -> output1
    input2 -> [FC2 with weights2] -> intermediate -> [FC3 with weights1] -> output2

  weights1 is shared between FC1 and FC3. weights2 is used only by FC2, which
  runs between the two consumers of weights1.
  """
  # 4 unique values per tensor for 2-bit LUT compression. Small values avoid
  # saturation in chained layers. Different row sums produce varied outputs.
  weights1_data = np.array([
      [-1, 0, 0, 1],
      [-1, 0, 1, 1],
      [-1, 1, 1, 1],
      [0, 1, 1, 1],
  ],
                           dtype=np.int8)
  weights1 = model_editor.Tensor(
      shape=(4, 4),
      dtype=tflite.TensorType.INT8,
      data=weights1_data,
      name="weights1",
      quantization=model_editor.Quantization(scales=1.0, zero_points=0),
  )

  weights2_data = np.array([
      [1, 1, 1, 1],
      [1, 1, 2, 2],
      [1, 2, 2, 3],
      [2, 2, 3, 3],
  ],
                           dtype=np.int8)
  weights2 = model_editor.Tensor(
      shape=(4, 4),
      dtype=tflite.TensorType.INT8,
      data=weights2_data,
      name="weights2",
      quantization=model_editor.Quantization(scales=1.0, zero_points=0),
  )

  # All tensors need matching quantization for FULLY_CONNECTED
  quant = model_editor.Quantization(scales=1.0, zero_points=0)

  input1 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="input1",
      quantization=quant,
  )
  input2 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="input2",
      quantization=quant,
  )
  output1 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="output1",
      quantization=quant,
  )
  intermediate = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="intermediate",
      quantization=quant,
  )
  output2 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="output2",
      quantization=quant,
  )

  model = model_editor.Model(subgraphs=[
      model_editor.Subgraph(
          tensors=[weights1, weights2],
          inputs=[input1, input2],
          outputs=[output1, output2],
          operators=[
              # FC1: uses weights1
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input1, weights1],
                  outputs=[output1],
              ),
              # FC2: uses weights2 (runs between FC1 and FC3)
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input2, weights2],
                  outputs=[intermediate],
              ),
              # FC3: uses weights1 (second consumer, after DECODE(weights2))
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[intermediate, weights1],
                  outputs=[output2],
              ),
          ],
      )
  ])
  return model.build()


class AltDecompressionMemoryTest(tf.test.TestCase):
  """Tests for alternate decompression memory with shared compressed tensors.

  These tests verify correct behavior when compressed tensors are shared
  between multiple operators and alternate decompression memory is enabled.
  """

  @unittest.expectedFailure
  def test_shared_compressed_tensor_with_alt_memory(self):
    """Verify correct results when a shared compressed tensor is used with alt
    decompression memory.

    This test uses a graph where a compressed tensor (weights1) is consumed by
    two operators (FC1 and FC3), with an intervening DECODE of a different
    compressed tensor (weights2) between them.

    The interpreter's alternate decompression memory has a limitation: each
    DECODE's Prepare resets the allocation offset to zero. This means all
    DECODE outputs are allocated at the same address, so they overwrite each
    other. A DECODE output can only be used until the next DECODE runs.

    To work around this limitation, the DECODE insertion code must insert a
    separate DECODE immediately before each consumer of a compressed tensor,
    rather than sharing one DECODE output among all consumers.

    This test is expected to fail because the current insertion code does not
    yet implement this workaround.
    """
    flatbuffer = _build_shared_weights_model()

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,  # weights1
            compression=[spec.LookUpTableCompression(index_bitwidth=2)],
        ),
        spec.Tensor(
            subgraph=0,
            tensor=1,  # weights2
            compression=[spec.LookUpTableCompression(index_bitwidth=2)],
        ),
    ]

    compressed_fb = compress.compress(flatbuffer, specs)

    # Run without alt decompression memory (baseline)
    interp_no_alt = runtime.Interpreter.from_bytes(bytes(compressed_fb))

    # Run with alt decompression memory
    interp_with_alt = runtime.Interpreter.from_bytes(
        bytes(compressed_fb),
        alt_decompression_memory_size=256,
    )

    test_input1 = np.array([[1, 1, 1, 1]], dtype=np.int8)
    test_input2 = np.array([[1, 1, 1, 1]], dtype=np.int8)

    interp_no_alt.set_input(test_input1, 0)
    interp_no_alt.set_input(test_input2, 1)
    interp_no_alt.invoke()
    expected1 = interp_no_alt.get_output(0)
    expected2 = interp_no_alt.get_output(1)

    interp_with_alt.set_input(test_input1, 0)
    interp_with_alt.set_input(test_input2, 1)
    interp_with_alt.invoke()
    actual1 = interp_with_alt.get_output(0)
    actual2 = interp_with_alt.get_output(1)

    self.assertAllEqual(expected1, actual1,
                        "Output 1 mismatch with alt decompression memory")
    self.assertAllEqual(expected2, actual2,
                        "Output 2 mismatch with alt decompression memory")


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
