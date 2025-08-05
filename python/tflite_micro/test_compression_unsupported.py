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
# ==============================================================================
"""Test compression metadata detection when compression is disabled."""

import os
import numpy as np
import tensorflow as tf
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro import compression


class CompressionDetectionTest(tf.test.TestCase):
  """Test compression metadata detection when compression is disabled."""

  def _create_test_model(self):
    """Create a simple quantized model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5, ), activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Convert to quantized TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
      for _ in range(10):
        yield [np.random.randn(1, 5).astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    return bytes(tflite_model) if isinstance(tflite_model,
                                             bytearray) else tflite_model

  def test_regular_model_loads_successfully(self):
    """Non-compressed models should load without issues."""
    model_data = self._create_test_model()
    interpreter = runtime.Interpreter.from_bytes(model_data)
    self.assertIsNotNone(interpreter)

  def test_compressed_model_raises_runtime_error(self):
    """Compressed models should raise RuntimeError when compression is disabled."""
    # Create and compress a model
    model_data = self._create_test_model()

    spec = (compression.SpecBuilder().add_tensor(
        subgraph=0, tensor=1).with_lut(index_bitwidth=4).build())

    compressed_model = compression.compress(model_data, spec)
    if isinstance(compressed_model, bytearray):
      compressed_model = bytes(compressed_model)

    # Should raise RuntimeError
    with self.assertRaises(RuntimeError):
      runtime.Interpreter.from_bytes(compressed_model)

  def test_can_load_regular_after_compressed_failure(self):
    """Verify we can still load regular models after compressed model fails."""
    model_data = self._create_test_model()

    # First try compressed model (should fail)
    spec = (compression.SpecBuilder().add_tensor(
        subgraph=0, tensor=1).with_lut(index_bitwidth=4).build())
    compressed_model = compression.compress(model_data, spec)

    with self.assertRaises(RuntimeError):
      runtime.Interpreter.from_bytes(bytes(compressed_model))

    # Then load regular model (should succeed)
    interpreter = runtime.Interpreter.from_bytes(model_data)
    self.assertIsNotNone(interpreter)


if __name__ == '__main__':
  # Set TF environment variables to suppress warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
  tf.test.main()
