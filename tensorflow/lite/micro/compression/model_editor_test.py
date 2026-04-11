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
"""Tests for model_editor module.
"""

import numpy as np
import tensorflow as tf
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression.model_editor import (
    Buffer, Model, Operator, OperatorCode, Quantization, Subgraph, Tensor)


class TestBasicModel(tf.test.TestCase):
  """Test basic model with tensors and operators."""

  @classmethod
  def setUpClass(cls):
    """Build model once for all tests in this class."""
    cls.input_data = np.array([[1, 2, 3, 4, 5]], dtype=np.int8)
    cls.weights_data = np.array([[1], [2], [3], [4], [5]], dtype=np.int8)

    cls.model = Model(
        description="Test model",
        subgraphs=[
            Subgraph(operators=[
                Operator(opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                         inputs=[
                             Tensor(shape=(1, 5),
                                    dtype=tflite.TensorType.INT8,
                                    data=cls.input_data,
                                    name="input"),
                             Tensor(shape=(5, 1),
                                    dtype=tflite.TensorType.INT8,
                                    data=cls.weights_data,
                                    name="weights")
                         ],
                         outputs=[
                             Tensor(shape=(1, 1),
                                    dtype=tflite.TensorType.INT8,
                                    name="output")
                         ])
            ])
        ])

    # Build the model to a flatbuffer byte array. This exercises the
    # model_editor's build path, which converts the high-level Model API
    # representation into the binary TFLite format.
    fb = cls.model.build()

    # Read the flatbuffer back through model_editor.read() to create a
    # loopback model. This exercises the read path, which parses the
    # flatbuffer and reconstructs a high-level Model representation. The
    # loopback model should be semantically equivalent to cls.model,
    # demonstrating that build() and read() are inverse operations.
    cls.loopback_model = model_editor.read(fb)

    # Parse the same flatbuffer using the low-level TFLite schema interface
    # (ModelT from schema_py_generated). This provides direct access to the
    # raw flatbuffer structure, allowing us to verify that model_editor
    # encodes data correctly at the binary level. We compare fb_model
    # (low-level) against loopback_model (high-level) to ensure both
    # representations are consistent.
    cls.fb_model = tflite.ModelT.InitFromPackedBuf(fb, 0)

  def test_description(self):
    """Verify model description is preserved through loopback."""
    self.assertEqual(self.fb_model.description, b"Test model")
    self.assertEqual(self.loopback_model.description, "Test model")

  def test_counts(self):
    """Verify subgraph, tensor, and operator counts."""
    self.assertEqual(len(self.fb_model.subgraphs), 1)
    self.assertEqual(len(self.loopback_model.subgraphs), 1)

    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    self.assertEqual(len(fb_sg.tensors), 3)
    self.assertEqual(len(loopback_sg.tensors), 3)

    self.assertEqual(len(fb_sg.operators), 1)
    self.assertEqual(len(loopback_sg.operators), 1)

  def test_tensor_names(self):
    """Verify tensor names are preserved."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Check that all expected tensor names are present
    fb_names = {t.name for t in fb_sg.tensors}
    self.assertEqual(fb_names, {b"input", b"weights", b"output"})

    loopback_names = {t.name for t in loopback_sg.tensors}
    self.assertEqual(loopback_names, {"input", "weights", "output"})

  def test_tensor_properties(self):
    """Verify tensor shapes and dtypes."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Input tensor
    input_fb = next(t for t in fb_sg.tensors if t.name == b"input")
    input_loopback = next(t for t in loopback_sg.tensors if t.name == "input")
    self.assertEqual(list(input_fb.shape), [1, 5])
    self.assertEqual(input_loopback.shape, (1, 5))
    self.assertEqual(input_fb.type, tflite.TensorType.INT8)
    self.assertEqual(input_loopback.dtype, tflite.TensorType.INT8)

    # Weights tensor
    weights_fb = next(t for t in fb_sg.tensors if t.name == b"weights")
    weights_loopback = next(t for t in loopback_sg.tensors
                            if t.name == "weights")
    self.assertEqual(list(weights_fb.shape), [5, 1])
    self.assertEqual(weights_loopback.shape, (5, 1))
    self.assertEqual(weights_fb.type, tflite.TensorType.INT8)
    self.assertEqual(weights_loopback.dtype, tflite.TensorType.INT8)

    # Output tensor
    output_fb = next(t for t in fb_sg.tensors if t.name == b"output")
    output_loopback = next(t for t in loopback_sg.tensors
                           if t.name == "output")
    self.assertEqual(list(output_fb.shape), [1, 1])
    self.assertEqual(output_loopback.shape, (1, 1))
    self.assertEqual(output_fb.type, tflite.TensorType.INT8)
    self.assertEqual(output_loopback.dtype, tflite.TensorType.INT8)

  def test_tensor_data(self):
    """Verify tensor data and buffer access."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Input tensor data
    input_buffer = self.fb_model.buffers[fb_sg.tensors[0].buffer]
    self.assertIsNotNone(input_buffer.data)
    self.assertEqual(bytes(input_buffer.data), self.input_data.tobytes())

    self.assertIsNotNone(loopback_sg.tensors[0].array)
    self.assertAllEqual(loopback_sg.tensors[0].array, self.input_data)

    # Weights tensor data
    weights_buffer = self.fb_model.buffers[fb_sg.tensors[1].buffer]
    self.assertIsNotNone(weights_buffer.data)
    self.assertEqual(bytes(weights_buffer.data), self.weights_data.tobytes())

    self.assertIsNotNone(loopback_sg.tensors[1].array)
    self.assertAllEqual(loopback_sg.tensors[1].array, self.weights_data)

    # Output tensor has no data
    self.assertEqual(fb_sg.tensors[2].buffer, 0)
    self.assertIsNone(loopback_sg.tensors[2].array)

  def test_buffer_allocation(self):
    """Verify buffer allocation and zero convention."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Exact buffer count: buffer 0 (empty) + input + weights = 3 total
    self.assertEqual(len(self.fb_model.buffers), 3)
    self.assertEqual(len(self.loopback_model.buffers), 3)

    # Buffer 0 is empty
    buffer_zero = self.fb_model.buffers[0]
    self.assertTrue(buffer_zero.data is None or len(buffer_zero.data) == 0)

    # Verify each buffer is referenced by exactly the expected tensor
    # Buffer 0 -> output tensor (no data)
    output_tensor = next(t for t in fb_sg.tensors if t.name == b"output")
    self.assertEqual(output_tensor.buffer, 0)

    # Buffer 1 and 2 -> input and weights (order may vary)
    input_tensor = next(t for t in fb_sg.tensors if t.name == b"input")
    weights_tensor = next(t for t in fb_sg.tensors if t.name == b"weights")
    self.assertNotEqual(input_tensor.buffer, 0)
    self.assertNotEqual(weights_tensor.buffer, 0)
    self.assertIn(input_tensor.buffer, [1, 2])
    self.assertIn(weights_tensor.buffer, [1, 2])

    # Tensors with data point to non-zero buffers in loopback model
    loopback_input_tensor = next(t for t in loopback_sg.tensors
                                 if t.name == "input")
    self.assertIsNotNone(loopback_input_tensor.buffer)
    self.assertIsNotNone(loopback_input_tensor.buffer.index)
    self.assertNotEqual(loopback_input_tensor.buffer.index, 0)
    self.assertEqual(len(loopback_input_tensor.buffer.data), 5)
    self.assertEqual(bytes(loopback_input_tensor.buffer.data),
                     self.input_data.tobytes())

  def test_operator_references(self):
    """Verify operators reference correct tensors."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Operator input/output references
    self.assertEqual(len(fb_sg.operators[0].inputs), 2)
    self.assertEqual([t.name for t in loopback_sg.operators[0].inputs],
                     ["input", "weights"])

    self.assertEqual(len(fb_sg.operators[0].outputs), 1)
    self.assertEqual([t.name for t in loopback_sg.operators[0].outputs],
                     ["output"])

    # Operator indices are in bounds
    num_tensors = len(fb_sg.tensors)
    for idx in list(fb_sg.operators[0].inputs) + list(
        fb_sg.operators[0].outputs):
      self.assertGreaterEqual(idx, 0)
      self.assertLess(idx, num_tensors)

  def test_operator_codes(self):
    """Verify operator code table is correctly populated."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    self.assertIsNotNone(self.fb_model.operatorCodes)
    self.assertEqual(len(self.fb_model.operatorCodes), 1)
    self.assertEqual(self.fb_model.operatorCodes[0].builtinCode,
                     tflite.BuiltinOperator.FULLY_CONNECTED)

    self.assertEqual(len(self.loopback_model.operator_codes), 1)
    self.assertIsNotNone(loopback_sg.operators[0].opcode_index)
    loopback_opcode = self.loopback_model.operator_codes[
        loopback_sg.operators[0].opcode_index]
    self.assertEqual(loopback_opcode.builtin_code,
                     tflite.BuiltinOperator.FULLY_CONNECTED)


class TestAdvancedModel(tf.test.TestCase):
  """Test multiple operators, custom ops, shared tensors, and mixed references."""

  @classmethod
  def setUpClass(cls):
    """Build model once for all tests in this class."""
    cls.input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int8)
    cls.weights_data = np.array(
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=np.int8)
    cls.bias_data = np.array([10], dtype=np.int8)
    # Int16 data to test endianness: values that will show byte order issues
    cls.int16_data = np.array([256, 512, 1024],
                              dtype=np.int16)  # 0x0100, 0x0200, 0x0400

    # Pre-declare shared tensor (output of FC, input to custom op)
    cls.hidden = Tensor(shape=(1, 1),
                        dtype=tflite.TensorType.INT8,
                        name="hidden")

    # Create explicit shared buffer to test buffer sharing between tensors
    cls.shared_buffer_data = np.array([100, 127], dtype=np.int8)
    cls.shared_buf = Buffer(data=cls.shared_buffer_data.tobytes())

    cls.model = Model(
        description="Advanced model",
        metadata={
            "version": b"1.0.0",
            "author": b"test_suite",
            "custom_data": bytes([0xDE, 0xAD, 0xBE, 0xEF])
        },
        subgraphs=[
            Subgraph(
                tensors=[
                    cls.hidden,  # Mixed: pre-declared shared tensor
                    # Int16 tensor to test endianness
                    Tensor(shape=(3, ),
                           dtype=tflite.TensorType.INT16,
                           data=cls.int16_data,
                           name="int16_tensor"),
                    # Two tensors sharing same buffer to test buffer deduplication
                    Tensor(shape=(2, ),
                           dtype=tflite.TensorType.INT8,
                           buffer=cls.shared_buf,
                           name="shared_buf_tensor1"),
                    Tensor(shape=(2, ),
                           dtype=tflite.TensorType.INT8,
                           buffer=cls.shared_buf,
                           name="shared_buf_tensor2")
                ],
                operators=[
                    # Multiple operators: FULLY_CONNECTED
                    Operator(
                        opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                        inputs=[
                            Tensor(shape=(1, 10),
                                   dtype=tflite.TensorType.INT8,
                                   data=cls.input_data,
                                   name="input"),
                            Tensor(shape=(10, 1),
                                   dtype=tflite.TensorType.INT8,
                                   data=cls.weights_data,
                                   name="weights")
                        ],
                        outputs=[cls.hidden
                                 ]  # Shared: reference to pre-declared
                    ),
                    # Custom operator
                    Operator(
                        opcode=tflite.BuiltinOperator.CUSTOM,
                        custom_code="MyCustomOp",
                        inputs=[cls.hidden],  # Shared: reuse hidden tensor
                        outputs=[
                            Tensor(shape=(1, 1),
                                   dtype=tflite.TensorType.INT8,
                                   name="processed")
                        ]),
                    # Multiple operators: ADD
                    Operator(
                        opcode=tflite.BuiltinOperator.ADD,
                        inputs=[
                            Tensor(
                                shape=(1, 1),
                                dtype=tflite.TensorType.INT8,
                                name="processed_ref"  # Mixed: inline tensor
                            ),
                            Tensor(shape=(1, 1),
                                   dtype=tflite.TensorType.INT8,
                                   data=cls.bias_data,
                                   name="bias")
                        ],
                        outputs=[
                            Tensor(shape=(1, 1),
                                   dtype=tflite.TensorType.INT8,
                                   name="output")
                        ])
                ])
        ])

    fb = cls.model.build()
    cls.loopback_model = model_editor.read(fb)
    cls.fb_model = tflite.ModelT.InitFromPackedBuf(fb, 0)

  def test_operator_counts(self):
    """Verify correct number of operators."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    self.assertEqual(len(fb_sg.operators), 3)
    self.assertEqual(len(loopback_sg.operators), 3)

  def test_operator_code_table(self):
    """Verify operator code table contains all operator types."""
    self.assertEqual(len(self.fb_model.operatorCodes), 3)
    self.assertEqual(len(self.loopback_model.operator_codes), 3)

    opcodes_fb = {op.builtinCode for op in self.fb_model.operatorCodes}
    self.assertIn(tflite.BuiltinOperator.FULLY_CONNECTED, opcodes_fb)
    self.assertIn(tflite.BuiltinOperator.CUSTOM, opcodes_fb)
    self.assertIn(tflite.BuiltinOperator.ADD, opcodes_fb)

    opcodes_loopback = {
        op.builtin_code
        for op in self.loopback_model.operator_codes
    }
    self.assertIn(tflite.BuiltinOperator.FULLY_CONNECTED, opcodes_loopback)
    self.assertIn(tflite.BuiltinOperator.CUSTOM, opcodes_loopback)
    self.assertIn(tflite.BuiltinOperator.ADD, opcodes_loopback)

  def test_custom_operator(self):
    """Verify custom operator code preservation."""
    loopback_sg = self.loopback_model.subgraphs[0]

    # Custom code in operator code table
    custom_opcode_fb = next(op for op in self.fb_model.operatorCodes
                            if op.builtinCode == tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(custom_opcode_fb.customCode, b"MyCustomOp")

    custom_opcode_loopback = next(
        op for op in self.loopback_model.operator_codes
        if op.builtin_code == tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(custom_opcode_loopback.custom_code, "MyCustomOp")

    # Custom operator references custom code
    custom_op_loopback = loopback_sg.operators[1]
    self.assertEqual(custom_op_loopback.opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(custom_op_loopback.custom_code, "MyCustomOp")

  def test_shared_tensor_references(self):
    """Verify tensors shared between operators."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Hidden tensor is at index 0 (pre-declared)
    self.assertEqual(fb_sg.tensors[0].name, b"hidden")
    self.assertEqual(loopback_sg.tensors[0].name, "hidden")

    # FC operator outputs to hidden
    self.assertEqual([t.name for t in loopback_sg.operators[0].outputs],
                     ["hidden"])

    # Custom operator inputs from hidden
    self.assertEqual([t.name for t in loopback_sg.operators[1].inputs],
                     ["hidden"])

    # Same Tensor object is referenced by both operators
    fc_output = loopback_sg.operators[0].outputs[0]
    custom_input = loopback_sg.operators[1].inputs[0]
    self.assertIs(fc_output, custom_input)

  def test_mixed_tensor_references(self):
    """Verify mix of pre-declared and inline tensors."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Total: hidden, int16_tensor, shared_buf_tensor1, shared_buf_tensor2 (pre-declared)
    # + input, weights, processed, processed_ref, bias, output (inline from operators)
    self.assertEqual(len(fb_sg.tensors), 10)
    self.assertEqual(len(loopback_sg.tensors), 10)

  def test_int16_endianness(self):
    """Verify int16 data is stored in little-endian byte order."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Find int16 tensor by name
    int16_tensor_fb = next(t for t in fb_sg.tensors
                           if t.name == b"int16_tensor")
    int16_tensor_loopback = next(t for t in loopback_sg.tensors
                                 if t.name == "int16_tensor")

    # Verify dtype
    self.assertEqual(int16_tensor_fb.type, tflite.TensorType.INT16)
    self.assertEqual(int16_tensor_loopback.dtype, tflite.TensorType.INT16)

    # Check flatbuffer buffer has correct little-endian bytes
    # For [256, 512, 1024] = [0x0100, 0x0200, 0x0400]
    # Little-endian bytes: [0x00, 0x01, 0x00, 0x02, 0x00, 0x04]
    int16_buffer_fb = self.fb_model.buffers[int16_tensor_fb.buffer]
    self.assertIsNotNone(int16_buffer_fb.data)
    expected_bytes = self.int16_data.astype(np.int16).astype('<i2').tobytes()
    self.assertEqual(bytes(int16_buffer_fb.data), expected_bytes)

    # Verify loopback reads it back correctly as int16 values
    self.assertIsNotNone(int16_tensor_loopback.array)
    self.assertAllEqual(int16_tensor_loopback.array, self.int16_data)

    # Verify buffer object provides correct bytes
    self.assertEqual(bytes(int16_tensor_loopback.buffer.data), expected_bytes)

  def test_metadata(self):
    """Verify metadata key-value pairs are preserved."""
    # Check flatbuffer metadata structure
    self.assertIsNotNone(self.fb_model.metadata)
    self.assertEqual(len(self.fb_model.metadata), 3)

    # Build name->buffer mapping from flatbuffer
    metadata_map_fb = {}
    for entry in self.fb_model.metadata:
      buffer_idx = entry.buffer
      self.assertLess(buffer_idx, len(self.fb_model.buffers))
      buffer = self.fb_model.buffers[buffer_idx]
      if buffer.data is not None:
        metadata_map_fb[entry.name] = bytes(buffer.data)

    # Verify flatbuffer metadata values
    self.assertEqual(metadata_map_fb[b"version"], b"1.0.0")
    self.assertEqual(metadata_map_fb[b"author"], b"test_suite")
    self.assertEqual(metadata_map_fb[b"custom_data"],
                     bytes([0xDE, 0xAD, 0xBE, 0xEF]))

    # Check loopback model metadata
    self.assertIsNotNone(self.loopback_model.metadata)
    self.assertEqual(len(self.loopback_model.metadata), 3)

    # Verify loopback metadata values (decoded from bytes)
    self.assertEqual(self.loopback_model.metadata["version"], b"1.0.0")
    self.assertEqual(self.loopback_model.metadata["author"], b"test_suite")
    self.assertEqual(self.loopback_model.metadata["custom_data"],
                     bytes([0xDE, 0xAD, 0xBE, 0xEF]))

  def test_buffer_allocation(self):
    """Verify no orphaned buffers and shared buffer deduplication."""
    fb_sg = self.fb_model.subgraphs[0]
    loopback_sg = self.loopback_model.subgraphs[0]

    # Collect all buffer references (from tensors and metadata)
    referenced_buffers = {0}  # Buffer 0 is special (always referenced)

    # Collect buffer references from tensors
    for tensor in fb_sg.tensors:
      referenced_buffers.add(tensor.buffer)

    # Collect buffer references from metadata
    for entry in self.fb_model.metadata:
      referenced_buffers.add(entry.buffer)

    # Verify no orphaned buffers (all buffers are referenced)
    for i in range(len(self.fb_model.buffers)):
      self.assertIn(
          i, referenced_buffers,
          f"Buffer {i} is orphaned (not referenced by any tensor or metadata)")

    # Verify shared buffer deduplication: two tensors share one buffer
    tensor1_fb = next(t for t in fb_sg.tensors
                      if t.name == b"shared_buf_tensor1")
    tensor2_fb = next(t for t in fb_sg.tensors
                      if t.name == b"shared_buf_tensor2")

    # Both tensors should point to the same buffer index
    self.assertEqual(tensor1_fb.buffer, tensor2_fb.buffer)
    self.assertNotEqual(tensor1_fb.buffer, 0)

    # Verify loopback preserves shared buffer (same Buffer object)
    tensor1_loopback = next(t for t in loopback_sg.tensors
                            if t.name == "shared_buf_tensor1")
    tensor2_loopback = next(t for t in loopback_sg.tensors
                            if t.name == "shared_buf_tensor2")

    self.assertIs(tensor1_loopback.buffer, tensor2_loopback.buffer)
    self.assertEqual(bytes(tensor1_loopback.buffer.data),
                     self.shared_buffer_data.tobytes())
    self.assertEqual(bytes(tensor2_loopback.buffer.data),
                     self.shared_buffer_data.tobytes())


class TestQuantization(tf.test.TestCase):
  """Test per-tensor and per-channel quantization parameters."""

  @classmethod
  def setUpClass(cls):
    """Build model once for all tests in this class."""
    # Per-channel quantization parameters
    cls.per_channel_scales = [0.1, 0.2, 0.3, 0.4]
    cls.per_channel_zeros = [0, 1, 2, 3]

    cls.model = Model(
        description="Quantization test model",
        subgraphs=[
            Subgraph(tensors=[
                # Per-tensor quantized tensor (single scale/zero_point)
                Tensor(shape=(1, 10),
                       dtype=tflite.TensorType.INT8,
                       data=np.ones((1, 10), dtype=np.int8),
                       name="per_tensor",
                       quantization=Quantization(scales=0.5, zero_points=10)),
                # Per-channel quantized tensor (array of scales/zero_points, axis)
                Tensor(shape=(4, 10),
                       dtype=tflite.TensorType.INT8,
                       data=np.ones((4, 10), dtype=np.int8),
                       name="per_channel",
                       quantization=Quantization(
                           scales=cls.per_channel_scales,
                           zero_points=cls.per_channel_zeros,
                           axis=0))
            ])
        ])

    fb = cls.model.build()
    cls.loopback_model = model_editor.read(fb)
    cls.fb_model = tflite.ModelT.InitFromPackedBuf(fb, 0)

  def test_per_tensor_quantization_flatbuffer(self):
    """Verify per-tensor quantization in flatbuffer encoding."""
    fb_sg = self.fb_model.subgraphs[0]

    tensor = next(t for t in fb_sg.tensors if t.name == b"per_tensor")
    self.assertIsNotNone(tensor.quantization)

    # Scale and zero_point encoded as single-element arrays
    self.assertIsNotNone(tensor.quantization.scale)
    self.assertEqual(len(tensor.quantization.scale), 1)
    self.assertEqual(tensor.quantization.scale[0], 0.5)

    self.assertIsNotNone(tensor.quantization.zeroPoint)
    self.assertEqual(len(tensor.quantization.zeroPoint), 1)
    self.assertEqual(tensor.quantization.zeroPoint[0], 10)

  def test_per_tensor_quantization_loopback(self):
    """Verify per-tensor quantization in loopback model."""
    loopback_sg = self.loopback_model.subgraphs[0]

    tensor = next(t for t in loopback_sg.tensors if t.name == "per_tensor")
    self.assertIsNotNone(tensor.quantization)

    # Read back as lists
    self.assertEqual(tensor.quantization.scales, [0.5])
    self.assertEqual(tensor.quantization.zero_points, [10])
    self.assertIsNone(tensor.quantization.axis)

  def test_per_channel_quantization_flatbuffer(self):
    """Verify per-channel quantization in flatbuffer encoding."""
    fb_sg = self.fb_model.subgraphs[0]

    tensor = next(t for t in fb_sg.tensors if t.name == b"per_channel")
    self.assertIsNotNone(tensor.quantization)

    # All scales encoded
    self.assertIsNotNone(tensor.quantization.scale)
    self.assertEqual(len(tensor.quantization.scale), 4)
    self.assertEqual(list(tensor.quantization.scale), self.per_channel_scales)

    # All zero_points encoded
    self.assertIsNotNone(tensor.quantization.zeroPoint)
    self.assertEqual(len(tensor.quantization.zeroPoint), 4)
    self.assertEqual(list(tensor.quantization.zeroPoint),
                     self.per_channel_zeros)

    # Axis encoded as quantizedDimension
    self.assertEqual(tensor.quantization.quantizedDimension, 0)

  def test_per_channel_quantization_loopback(self):
    """Verify per-channel quantization in loopback model."""
    loopback_sg = self.loopback_model.subgraphs[0]

    tensor = next(t for t in loopback_sg.tensors if t.name == "per_channel")
    self.assertIsNotNone(tensor.quantization)

    # Read back as lists
    self.assertEqual(tensor.quantization.scales, self.per_channel_scales)
    self.assertEqual(tensor.quantization.zero_points, self.per_channel_zeros)
    self.assertEqual(tensor.quantization.axis, 0)


class TestReadModifyWrite(tf.test.TestCase):
  """Test read-modify-write workflows."""

  @classmethod
  def setUpClass(cls):
    """Create a simple base model for modification tests."""
    cls.original_data = np.array([[1, 2, 3]], dtype=np.int8)
    cls.model = Model(
        description="Base model",
        metadata={"original": b"metadata"},
        subgraphs=[
            Subgraph(tensors=[
                Tensor(shape=(1, 3),
                       dtype=tflite.TensorType.INT8,
                       data=cls.original_data,
                       name="weights"),
                Tensor(
                    shape=(1, 3), dtype=tflite.TensorType.INT8, name="input"),
                Tensor(
                    shape=(1, 3), dtype=tflite.TensorType.INT8, name="output")
            ])
        ])

    cls.fb = cls.model.build()

  def test_modify_tensor_data(self):
    """Read model, modify tensor data, write back, verify."""
    # Read the model
    model2 = model_editor.read(self.fb)

    # Modify tensor data using array setter (high-level API)
    weights_tensor = next(t for t in model2.subgraphs[0].tensors
                          if t.name == "weights")
    new_data = np.array([[10, 20, 30]], dtype=np.int8)
    weights_tensor.array = new_data  # Uses array setter

    # Build modified model
    fb2 = model2.build()

    # Read back and verify modification
    model3 = model_editor.read(fb2)
    modified_weights = next(t for t in model3.subgraphs[0].tensors
                            if t.name == "weights")
    self.assertAllEqual(modified_weights.array, new_data)

    # Verify other tensors unchanged
    self.assertEqual(len(model3.subgraphs[0].tensors), 3)

  def test_add_tensor_and_operator(self):
    """Read model, add new tensor and operator, write back, verify."""
    # Read the model
    model2 = model_editor.read(self.fb)
    sg = model2.subgraphs[0]

    # Get existing tensors
    input_tensor = next(t for t in sg.tensors if t.name == "input")
    output_tensor = next(t for t in sg.tensors if t.name == "output")

    # Add new tensor using imperative API
    new_weights = np.array([[5, 10, 15]], dtype=np.int8)
    new_weights_tensor = sg.add_tensor(shape=(1, 3),
                                       dtype=tflite.TensorType.INT8,
                                       data=new_weights,
                                       name="new_weights")

    # Add new operator using imperative API
    sg.add_operator(opcode=tflite.BuiltinOperator.ADD,
                    inputs=[input_tensor, new_weights_tensor],
                    outputs=[output_tensor])

    # Build modified model
    fb2 = model2.build()

    # Read back and verify additions
    model3 = model_editor.read(fb2)
    sg3 = model3.subgraphs[0]

    # Verify tensor was added
    self.assertEqual(len(sg3.tensors), 4)
    added_tensor = next(t for t in sg3.tensors if t.name == "new_weights")
    self.assertIsNotNone(added_tensor)
    self.assertAllEqual(added_tensor.array, new_weights)

    # Verify operator was added
    self.assertEqual(len(sg3.operators), 1)
    added_op = sg3.operators[0]
    self.assertEqual([t.name for t in added_op.inputs],
                     ["input", "new_weights"])
    self.assertEqual([t.name for t in added_op.outputs], ["output"])

  def test_modify_metadata(self):
    """Read model, modify metadata, write back, verify."""
    # Read the model
    model2 = model_editor.read(self.fb)

    # Modify existing metadata
    model2.metadata["original"] = b"modified_metadata"

    # Add new metadata
    model2.metadata["new_key"] = b"new_value"

    # Build modified model
    fb2 = model2.build()

    # Read back and verify modifications
    model3 = model_editor.read(fb2)

    self.assertEqual(len(model3.metadata), 2)
    self.assertEqual(model3.metadata["original"], b"modified_metadata")
    self.assertEqual(model3.metadata["new_key"], b"new_value")


class TestSubgraphInputsOutputs(tf.test.TestCase):
  """Test subgraph inputs and outputs are set correctly."""

  def test_subgraph_inputs_outputs_set(self):
    """Verify subgraph inputs/outputs are set in the flatbuffer."""
    input_t = Tensor(shape=(1, 4), dtype=tflite.TensorType.INT8, name="input")
    output_t = Tensor(shape=(1, 4),
                      dtype=tflite.TensorType.INT8,
                      name="output")
    weights = Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        data=np.array([[1, 2, 3, 4]] * 4, dtype=np.int8),
        name="weights",
    )

    model = Model(subgraphs=[
        Subgraph(
            tensors=[weights],
            inputs=[input_t],
            outputs=[output_t],
            operators=[
                Operator(
                    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                    inputs=[input_t, weights],
                    outputs=[output_t],
                )
            ],
        )
    ])

    fb = model.build()
    fb_model = tflite.ModelT.InitFromPackedBuf(fb, 0)
    fb_sg = fb_model.subgraphs[0]

    # Verify inputs/outputs are set (as tensor indices)
    self.assertEqual(len(fb_sg.inputs), 1)
    self.assertEqual(len(fb_sg.outputs), 1)

    # Verify indices point to correct tensors
    input_idx = fb_sg.inputs[0]
    output_idx = fb_sg.outputs[0]
    self.assertEqual(fb_sg.tensors[input_idx].name, b"input")
    self.assertEqual(fb_sg.tensors[output_idx].name, b"output")

  def test_subgraph_inputs_outputs_loopback(self):
    """Verify inputs/outputs survive read/build loopback."""
    input_t = Tensor(shape=(1, 4), dtype=tflite.TensorType.INT8, name="input")
    output_t = Tensor(shape=(1, 4),
                      dtype=tflite.TensorType.INT8,
                      name="output")
    weights = Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        data=np.array([[1, 2, 3, 4]] * 4, dtype=np.int8),
        name="weights",
    )

    model = Model(subgraphs=[
        Subgraph(
            tensors=[weights],
            inputs=[input_t],
            outputs=[output_t],
            operators=[
                Operator(
                    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                    inputs=[input_t, weights],
                    outputs=[output_t],
                )
            ],
        )
    ])

    fb = model.build()
    loopback = model_editor.read(fb)
    sg = loopback.subgraphs[0]

    # Verify high-level inputs/outputs are populated
    self.assertEqual(len(sg.inputs), 1)
    self.assertEqual(len(sg.outputs), 1)
    self.assertEqual(sg.inputs[0].name, "input")
    self.assertEqual(sg.outputs[0].name, "output")

  def test_tensor_by_name_not_found_raises(self):
    """tensor_by_name raises KeyError when name not found."""
    model = Model(subgraphs=[
        Subgraph(tensors=[
            Tensor(shape=(4, ), dtype=tflite.TensorType.INT8, name="exists")
        ])
    ])

    with self.assertRaises(KeyError):
      model.subgraphs[0].tensor_by_name("nonexistent")


class TestReadEdgeCases(tf.test.TestCase):
  """Test model_editor.read() with edge cases from real-world models.

  These tests construct models using the low-level TFLite schema to create
  edge cases that may not be producible via model_editor.build(), but can
  appear in models from other sources (e.g., TFLite converter).
  """

  def _build_model_with_schema(self, model_t):
    """Build a flatbuffer from a ModelT using the low-level schema."""
    import flatbuffers
    builder = flatbuffers.Builder(1024)
    builder.Finish(model_t.Pack(builder))
    return bytes(builder.Output())

  def test_read_scalar_tensor(self):
    """Verify read() handles tensors with None shape (scalars).

    Some TFLite models have scalar tensors where shape is None rather than
    an empty list. This can occur with constant scalars produced by certain
    converters.
    """
    # Build a minimal model with a scalar tensor (shape=None)
    model_t = tflite.ModelT()
    model_t.version = 3

    # Buffer 0 is always empty, buffer 1 holds scalar data
    buf0 = tflite.BufferT()
    buf0.data = []
    buf1 = tflite.BufferT()
    buf1.data = [42]  # Single byte scalar value

    model_t.buffers = [buf0, buf1]

    # Create operator code
    opcode = tflite.OperatorCodeT()
    opcode.builtinCode = tflite.BuiltinOperator.ADD
    model_t.operatorCodes = [opcode]

    # Create subgraph with scalar tensor
    sg = tflite.SubGraphT()

    # Tensor with shape=None (scalar)
    scalar_tensor = tflite.TensorT()
    scalar_tensor.name = b"scalar"
    scalar_tensor.type = tflite.TensorType.INT8
    scalar_tensor.buffer = 1
    scalar_tensor.shape = None  # This is the edge case

    # Normal tensor for comparison
    normal_tensor = tflite.TensorT()
    normal_tensor.name = b"normal"
    normal_tensor.type = tflite.TensorType.INT8
    normal_tensor.buffer = 0
    normal_tensor.shape = [1, 4]

    sg.tensors = [scalar_tensor, normal_tensor]
    sg.inputs = [1]
    sg.outputs = [1]
    sg.operators = []

    model_t.subgraphs = [sg]

    # Build and read
    fb = self._build_model_with_schema(model_t)
    model = model_editor.read(fb)

    # Verify scalar tensor was read with empty shape tuple
    self.assertEqual(model.subgraphs[0].tensors[0].shape, ())
    self.assertEqual(model.subgraphs[0].tensors[0].name, "scalar")

    # Verify normal tensor shape is preserved
    self.assertEqual(model.subgraphs[0].tensors[1].shape, (1, 4))

  def test_read_operator_with_empty_inputs(self):
    """Verify read() handles operators with None inputs/outputs.

    Some operators (e.g., certain control flow or custom ops) may have
    empty input or output lists represented as None in the flatbuffer.
    """
    model_t = tflite.ModelT()
    model_t.version = 3

    buf0 = tflite.BufferT()
    buf0.data = []
    model_t.buffers = [buf0]

    # Custom op that might have unusual input/output patterns
    opcode = tflite.OperatorCodeT()
    opcode.builtinCode = tflite.BuiltinOperator.CUSTOM
    opcode.customCode = b"NoInputOp"
    model_t.operatorCodes = [opcode]

    sg = tflite.SubGraphT()

    # Single output tensor
    output_tensor = tflite.TensorT()
    output_tensor.name = b"output"
    output_tensor.type = tflite.TensorType.INT8
    output_tensor.buffer = 0
    output_tensor.shape = [1]

    sg.tensors = [output_tensor]
    sg.inputs = []
    sg.outputs = [0]

    # Operator with None inputs (edge case)
    op = tflite.OperatorT()
    op.opcodeIndex = 0
    op.inputs = None  # This is the edge case
    op.outputs = [0]

    sg.operators = [op]
    model_t.subgraphs = [sg]

    # Build and read
    fb = self._build_model_with_schema(model_t)
    model = model_editor.read(fb)

    # Verify operator was read with empty inputs list
    self.assertEqual(len(model.subgraphs[0].operators), 1)
    self.assertEqual(model.subgraphs[0].operators[0].inputs, [])
    self.assertEqual(len(model.subgraphs[0].operators[0].outputs), 1)

  def test_read_operator_with_empty_outputs(self):
    """Verify read() handles operators with None outputs.

    Similar to empty inputs, some operators may have None outputs.
    """
    model_t = tflite.ModelT()
    model_t.version = 3

    buf0 = tflite.BufferT()
    buf0.data = []
    model_t.buffers = [buf0]

    opcode = tflite.OperatorCodeT()
    opcode.builtinCode = tflite.BuiltinOperator.CUSTOM
    opcode.customCode = b"NoOutputOp"
    model_t.operatorCodes = [opcode]

    sg = tflite.SubGraphT()

    input_tensor = tflite.TensorT()
    input_tensor.name = b"input"
    input_tensor.type = tflite.TensorType.INT8
    input_tensor.buffer = 0
    input_tensor.shape = [1]

    sg.tensors = [input_tensor]
    sg.inputs = [0]
    sg.outputs = []

    # Operator with None outputs (edge case)
    op = tflite.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = None  # This is the edge case

    sg.operators = [op]
    model_t.subgraphs = [sg]

    fb = self._build_model_with_schema(model_t)
    model = model_editor.read(fb)

    self.assertEqual(len(model.subgraphs[0].operators), 1)
    self.assertEqual(len(model.subgraphs[0].operators[0].inputs), 1)
    self.assertEqual(model.subgraphs[0].operators[0].outputs, [])

  def test_int64_tensor(self):
    """Verify INT64 tensors are correctly handled."""
    model_t = tflite.ModelT()
    model_t.version = 3

    buf0 = tflite.BufferT()
    buf0.data = []
    buf1 = tflite.BufferT()
    # INT64 data: [1, 2, 3, 4] as little-endian 8-byte values
    int64_data = np.array([1, 2, 3, 4], dtype=np.int64)
    buf1.data = list(int64_data.tobytes())

    model_t.buffers = [buf0, buf1]

    opcode = tflite.OperatorCodeT()
    opcode.builtinCode = tflite.BuiltinOperator.ADD
    model_t.operatorCodes = [opcode]

    sg = tflite.SubGraphT()
    tensor = tflite.TensorT()
    tensor.name = b"int64_tensor"
    tensor.type = tflite.TensorType.INT64
    tensor.buffer = 1
    tensor.shape = [4]

    sg.tensors = [tensor]
    sg.inputs = [0]
    sg.outputs = [0]
    sg.operators = []
    model_t.subgraphs = [sg]

    fb = self._build_model_with_schema(model_t)
    model = model_editor.read(fb)

    t = model.subgraphs[0].tensors[0]
    self.assertEqual(t.dtype, tflite.TensorType.INT64)
    self.assertAllEqual(t.array, int64_data)


class TestFieldPreservation(tf.test.TestCase):
  """Test that schema fields are preserved during read-modify-write.

  These tests verify that fields not explicitly handled by model_editor
  are still preserved when reading a model, modifying it, and writing
  it back. This catches regressions where adding wrapper classes might
  accidentally drop fields.
  """

  def _build_model_with_schema(self, model_t):
    """Build a flatbuffer from a ModelT using the low-level schema."""
    import flatbuffers
    builder = flatbuffers.Builder(1024)
    builder.Finish(model_t.Pack(builder))
    return bytes(builder.Output())

  def _create_base_model(self):
    """Create a minimal valid model for testing."""
    model_t = tflite.ModelT()
    model_t.version = 3
    model_t.description = b"test"

    buf0 = tflite.BufferT()
    buf0.data = []
    buf1 = tflite.BufferT()
    buf1.data = [1, 2, 3, 4]
    model_t.buffers = [buf0, buf1]

    opcode = tflite.OperatorCodeT()
    opcode.builtinCode = tflite.BuiltinOperator.ADD
    model_t.operatorCodes = [opcode]

    sg = tflite.SubGraphT()

    t0 = tflite.TensorT()
    t0.name = b"input"
    t0.type = tflite.TensorType.INT8
    t0.buffer = 1
    t0.shape = [4]

    t1 = tflite.TensorT()
    t1.name = b"output"
    t1.type = tflite.TensorType.INT8
    t1.buffer = 0
    t1.shape = [4]

    sg.tensors = [t0, t1]
    sg.inputs = [0]
    sg.outputs = [1]

    op = tflite.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [1]
    sg.operators = [op]

    model_t.subgraphs = [sg]
    return model_t

  def test_tensor_is_variable_preserved(self):
    """Verify Tensor.isVariable is preserved through read-modify-write."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].tensors[0].isVariable = True

    fb = self._build_model_with_schema(model_t)

    # Read, modify, write
    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    # Verify field preserved
    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertTrue(model_t2.subgraphs[0].tensors[0].isVariable)

  def test_tensor_shape_signature_preserved(self):
    """Verify Tensor.shapeSignature is preserved through read-modify-write."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].tensors[0].shapeSignature = [-1, 4]

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(list(model_t2.subgraphs[0].tensors[0].shapeSignature),
                     [-1, 4])

  def test_operator_builtin_options_preserved(self):
    """Verify Operator.builtinOptions is preserved through read-modify-write."""
    model_t = self._create_base_model()

    # Use ADD operator with AddOptions (must also set builtinOptionsType for union)
    add_options = tflite.AddOptionsT()
    add_options.fusedActivationFunction = tflite.ActivationFunctionType.RELU
    model_t.subgraphs[0].operators[0].builtinOptions = add_options
    model_t.subgraphs[0].operators[
        0].builtinOptionsType = tflite.BuiltinOptions.AddOptions

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertIsNotNone(model_t2.subgraphs[0].operators[0].builtinOptions)
    self.assertEqual(
        model_t2.subgraphs[0].operators[0].builtinOptions.
        fusedActivationFunction, tflite.ActivationFunctionType.RELU)

  def test_operator_custom_options_preserved(self):
    """Verify Operator.customOptions is preserved through read-modify-write."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].operators[0].customOptions = [0xDE, 0xAD, 0xBE, 0xEF]

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(list(model_t2.subgraphs[0].operators[0].customOptions),
                     [0xDE, 0xAD, 0xBE, 0xEF])

  def test_operator_intermediates_preserved(self):
    """Verify Operator.intermediates is preserved through read-modify-write."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].operators[0].intermediates = [0, 1]

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(list(model_t2.subgraphs[0].operators[0].intermediates),
                     [0, 1])

  def test_operator_debug_metadata_index_preserved(self):
    """Verify Operator.debugMetadataIndex is preserved through read-modify-write."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].operators[0].debugMetadataIndex = 7

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(model_t2.subgraphs[0].operators[0].debugMetadataIndex, 7)

  def test_operator_code_deprecated_builtin_code_preserved(self):
    """Verify OperatorCode.deprecatedBuiltinCode is preserved."""
    model_t = self._create_base_model()
    # Set deprecated code to a value different from the new builtin code
    model_t.operatorCodes[0].deprecatedBuiltinCode = 42

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(model_t2.operatorCodes[0].deprecatedBuiltinCode, 42)

  def test_subgraph_debug_metadata_index_preserved(self):
    """Verify SubGraph.debugMetadataIndex is preserved."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].debugMetadataIndex = 5

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(model_t2.subgraphs[0].debugMetadataIndex, 5)

  def test_model_version_preserved(self):
    """Verify Model.version is preserved (not hardcoded to 3)."""
    model_t = self._create_base_model()
    model_t.version = 42

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertEqual(model_t2.version, 42)

  def test_model_signature_defs_preserved(self):
    """Verify Model.signatureDefs is preserved."""
    model_t = self._create_base_model()

    sig_def = tflite.SignatureDefT()
    sig_def.signatureKey = b"serving_default"
    sig_def.subgraphIndex = 0
    model_t.signatureDefs = [sig_def]

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertIsNotNone(model_t2.signatureDefs)
    self.assertEqual(len(model_t2.signatureDefs), 1)
    self.assertEqual(model_t2.signatureDefs[0].signatureKey,
                     b"serving_default")

  def test_quantization_min_max_preserved(self):
    """Verify QuantizationParameters.min/max are preserved."""
    model_t = self._create_base_model()

    quant = tflite.QuantizationParametersT()
    quant.scale = [0.5]
    quant.zeroPoint = [128]
    quant.min = [0.0]
    quant.max = [1.0]
    model_t.subgraphs[0].tensors[0].quantization = quant

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    quant2 = model_t2.subgraphs[0].tensors[0].quantization
    self.assertIsNotNone(quant2)
    self.assertEqual(list(quant2.min), [0.0])
    self.assertEqual(list(quant2.max), [1.0])

  def test_tensor_sparsity_preserved(self):
    """Verify Tensor.sparsity is preserved through read-modify-write."""
    model_t = self._create_base_model()

    sparsity = tflite.SparsityParametersT()
    sparsity.traversalOrder = [0, 1]
    sparsity.blockMap = [0]
    model_t.subgraphs[0].tensors[0].sparsity = sparsity

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    sparsity2 = model_t2.subgraphs[0].tensors[0].sparsity
    self.assertIsNotNone(sparsity2)
    self.assertEqual(list(sparsity2.traversalOrder), [0, 1])
    self.assertEqual(list(sparsity2.blockMap), [0])

  def test_tensor_has_rank_preserved(self):
    """Verify Tensor.hasRank is preserved through read-modify-write."""
    model_t = self._create_base_model()
    model_t.subgraphs[0].tensors[0].hasRank = True

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    self.assertTrue(model_t2.subgraphs[0].tensors[0].hasRank)

  def test_operator_builtin_options_2_preserved(self):
    """Verify Operator.builtinOptions2 is preserved through read-modify-write."""
    model_t = self._create_base_model()

    options2 = tflite.StablehloConcatenateOptionsT()
    options2.dimension = 42
    model_t.subgraphs[0].operators[0].builtinOptions2 = options2
    model_t.subgraphs[0].operators[0].builtinOptions2Type = (
        tflite.BuiltinOptions2.StablehloConcatenateOptions)

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    options2_out = model_t2.subgraphs[0].operators[0].builtinOptions2
    self.assertIsNotNone(options2_out)
    self.assertEqual(options2_out.dimension, 42)

  def test_quantization_axis_preserved(self):
    """Verify QuantizationParameters.quantizedDimension is preserved."""
    model_t = self._create_base_model()

    quant = tflite.QuantizationParametersT()
    quant.scale = [0.5, 0.25]
    quant.zeroPoint = [0, 0]
    quant.quantizedDimension = 1
    model_t.subgraphs[0].tensors[0].quantization = quant

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    quant2 = model_t2.subgraphs[0].tensors[0].quantization
    self.assertIsNotNone(quant2)
    self.assertEqual(quant2.quantizedDimension, 1)

  def test_quantization_zero_point_preserved(self):
    """Verify QuantizationParameters.zeroPoint is preserved."""
    model_t = self._create_base_model()

    quant = tflite.QuantizationParametersT()
    quant.scale = [0.5]
    quant.zeroPoint = [128]
    model_t.subgraphs[0].tensors[0].quantization = quant

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    quant2 = model_t2.subgraphs[0].tensors[0].quantization
    self.assertIsNotNone(quant2)
    self.assertEqual(list(quant2.zeroPoint), [128])

  def test_quantization_zero_point_not_expanded(self):
    """Single zeroPoint with multiple scales is preserved as-is.

    TFLite converter optimizes by storing single zeroPoint when all channels
    have the same zero point. This must be preserved, not expanded.
    """
    model_t = self._create_base_model()

    quant = tflite.QuantizationParametersT()
    quant.scale = [0.5, 0.25, 0.125, 0.0625]  # 4 scales
    quant.zeroPoint = [128]  # Single zero point (converter optimization)
    quant.quantizedDimension = 0
    model_t.subgraphs[0].tensors[0].quantization = quant

    fb = self._build_model_with_schema(model_t)

    model = model_editor.read(fb)
    model.description = "modified"
    fb2 = model.build()

    model_t2 = tflite.ModelT.InitFromPackedBuf(fb2, 0)
    quant2 = model_t2.subgraphs[0].tensors[0].quantization
    self.assertIsNotNone(quant2)
    # Should still be single-element, not expanded to 4
    self.assertEqual(len(quant2.zeroPoint), 1)
    self.assertEqual(quant2.zeroPoint[0], 128)


if __name__ == "__main__":
  tf.test.main()
