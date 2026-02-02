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
"""Integration tests for compression using proprietary models.

These tests verify that compressed models produce correct inference results
when run through the TFLM Python interpreter. Tests compress models and
compare outputs against uncompressed originals using random inputs.

This test is tagged `manual` and requires a path to a directory containing
.tflite model files.

Usage:
    bazel test //tensorflow/lite/micro/compression:proprietary_integration_test \
        --//:with_compression \
        --test_arg=/path/to/models

Required files:
    Each model requires a compression spec file:
        model.spec.yaml  (replacing .tflite extension)

    See spec.py for the YAML format. Example:
        tensors:
          - subgraph: 0
            tensor: 2
            compression:
              - lut:
                  index_bitwidth: 4

Optional files:
    model.config.json  (replacing .tflite extension)
        Tolerance overrides: {"rtol": 1e-5, "atol": 1e-6}
        Default is exact match (rtol=0, atol=0).
"""

import glob
import json
import os
import sys
import unittest

import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


def _dtype_to_numpy(dtype: tflite.TensorType) -> np.dtype:
  """Convert TFLite dtype to numpy dtype."""
  type_map = {
      tflite.TensorType.INT8: np.int8,
      tflite.TensorType.INT16: np.int16,
      tflite.TensorType.INT32: np.int32,
      tflite.TensorType.INT64: np.int64,
      tflite.TensorType.UINT8: np.uint8,
      tflite.TensorType.UINT16: np.uint16,
      tflite.TensorType.UINT32: np.uint32,
      tflite.TensorType.FLOAT16: np.float16,
      tflite.TensorType.FLOAT32: np.float32,
      tflite.TensorType.FLOAT64: np.float64,
      tflite.TensorType.BOOL: np.bool_,
  }
  return type_map.get(dtype, np.uint8)


class ProprietaryModelTest(tf.test.TestCase):
  """Integration tests using proprietary models."""

  # Parsed from command line in main()
  models_dir = None

  @classmethod
  def setUpClass(cls):
    if not cls.models_dir:
      raise unittest.SkipTest(
          "No models directory provided. "
          "Usage: bazel test ... --test_arg=/path/to/models")

    cls.model_paths = sorted(
        glob.glob(os.path.join(cls.models_dir, '*.tflite')))
    if not cls.model_paths:
      raise unittest.SkipTest(f"No .tflite files found in {cls.models_dir}")

  def test_all_models(self):
    """Run compression test on each discovered model."""
    for model_path in self.model_paths:
      with self.subTest(model=os.path.basename(model_path)):
        self._test_model_compression(model_path)

  def _test_model_compression(self, model_path):
    """Test that a compressed model produces same outputs as original."""
    with open(model_path, 'rb') as f:
      flatbuffer = f.read()

    # Load compression spec from sidecar file
    specs = self._load_compression_spec(model_path)

    # Load tolerance config
    rtol, atol = self._load_tolerance(model_path)

    # Compress the model
    compressed_fb = compress.compress(flatbuffer, specs)

    # Create interpreters
    original_interp = runtime.Interpreter.from_bytes(bytes(flatbuffer))
    compressed_interp = runtime.Interpreter.from_bytes(bytes(compressed_fb))

    # Generate random inputs and compare outputs
    np.random.seed(42)
    model = model_editor.read(flatbuffer)
    sg = model.subgraphs[0]

    for trial in range(5):
      # Set inputs
      for i, input_tensor in enumerate(sg.inputs):
        test_input = self._generate_input(input_tensor)
        original_interp.set_input(test_input, i)
        compressed_interp.set_input(test_input, i)

      # Run inference
      original_interp.invoke()
      compressed_interp.invoke()

      # Compare outputs
      for i in range(len(sg.outputs)):
        expected = original_interp.get_output(i)
        actual = compressed_interp.get_output(i)
        self._compare_outputs(expected, actual, rtol, atol,
                              f"trial {trial}, output {i}")

  def _generate_input(self, tensor):
    """Generate random input respecting tensor dtype."""
    shape = tensor.shape
    dtype = _dtype_to_numpy(tensor.dtype)

    if np.issubdtype(dtype, np.floating):
      return np.random.uniform(-1.0, 1.0, shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
      info = np.iinfo(dtype)
      return np.random.randint(info.min, info.max + 1, shape, dtype=dtype)
    elif dtype == np.bool_:
      return np.random.choice([False, True], shape)
    return np.zeros(shape, dtype=dtype)

  def _load_compression_spec(self, model_path):
    """Load compression spec from sidecar YAML file.

    Raises:
      FileNotFoundError: If no spec file is found.
    """
    spec_path = model_path.replace('.tflite', '.spec.yaml')
    if os.path.exists(spec_path):
      with open(spec_path) as f:
        return spec.parse_yaml(f.read())

    raise FileNotFoundError(
        f"No compression spec file found for {model_path}. "
        f"Expected: {spec_path}")

  def _load_tolerance(self, model_path):
    """Load tolerance from sidecar config if present.

    Returns (0, 0) for exact match if no config file exists.
    """
    config_path = model_path.replace('.tflite', '.config.json')
    if os.path.exists(config_path):
      with open(config_path) as f:
        config = json.load(f)
      return config.get('rtol', 0), config.get('atol', 0)
    return 0, 0

  def _compare_outputs(self, expected, actual, rtol, atol, context=""):
    """Compare outputs with optional tolerance."""
    msg = f"Output mismatch ({context})" if context else "Output mismatch"
    if rtol == 0 and atol == 0:
      np.testing.assert_array_equal(expected, actual, err_msg=msg)
    else:
      np.testing.assert_allclose(expected,
                                 actual,
                                 rtol=rtol,
                                 atol=atol,
                                 err_msg=msg)


if __name__ == "__main__":
  # Suppress TF C++ info/debug logs (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  # Disable oneDNN to avoid non-deterministic floating point results
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

  # Parse models directory from args, then strip it so tf.test doesn't see it
  for arg in sys.argv[1:]:
    if not arg.startswith('-') and os.path.isdir(arg):
      ProprietaryModelTest.models_dir = arg
      sys.argv.remove(arg)
      break

  tf.test.main()