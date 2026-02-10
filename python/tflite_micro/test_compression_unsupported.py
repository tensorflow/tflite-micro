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
"""Test legacy compression metadata detection when compression is disabled."""

import os
import numpy as np
import tensorflow as tf
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.compression import model_editor


def _create_test_model():
  """Create a simple quantized model for testing."""
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
      tf.keras.layers.Dense(5, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

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


def _inject_compression_metadata(model_data):
  """Inject raw COMPRESSION_METADATA into a model's flatbuffer metadata.

  This simulates a legacy-compressed model (one that uses the
  COMPRESSION_METADATA metadata entry and kernel-level decompression) without
  going through compress(), which now produces DECODE-based output.
  """
  model = model_editor.read(model_data)
  model.metadata["COMPRESSION_METADATA"] = b"\x00"
  return bytes(model.build())


class LegacyCompressionDetectionTest(tf.test.TestCase):
  """Test that legacy COMPRESSION_METADATA is rejected without the flag."""

  def test_regular_model_loads_successfully(self):
    """Non-compressed models should load without issues."""
    model_data = _create_test_model()
    interpreter = runtime.Interpreter.from_bytes(model_data)
    self.assertIsNotNone(interpreter)

  def test_legacy_compressed_model_raises_runtime_error(self):
    """Models with COMPRESSION_METADATA should raise RuntimeError."""
    model_data = _create_test_model()
    legacy_model = _inject_compression_metadata(model_data)

    with self.assertRaises(RuntimeError):
      runtime.Interpreter.from_bytes(legacy_model)

  def test_can_load_regular_after_legacy_failure(self):
    """Verify regular models still load after a legacy-compressed failure."""
    model_data = _create_test_model()
    legacy_model = _inject_compression_metadata(model_data)

    with self.assertRaises(RuntimeError):
      runtime.Interpreter.from_bytes(legacy_model)

    interpreter = runtime.Interpreter.from_bytes(model_data)
    self.assertIsNotNone(interpreter)


if __name__ == '__main__':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
  tf.test.main()
