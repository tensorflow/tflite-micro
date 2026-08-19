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
        --test_arg=--models-dir=/path/to/models

Required files:
    Each model requires a compression spec file:
        model.spec.yaml  (replacing .tflite extension)

    See spec.py for the YAML format.

Optional files:
    model.config.yaml  (replacing .tflite extension)
        Comparison tolerances. Example:
            rtol: 1.0e-6
            atol: 1.0e-6
        An output element passes when:
            abs(actual - expected) <= atol + rtol * abs(expected)
        rtol scales with the expected value's magnitude; atol is an
        absolute floor for values near zero. Without this file,
        outputs must match exactly.
"""

import argparse
import glob
import os
import sys
import unittest

import yaml

from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import verify


class ProprietaryModelTest(unittest.TestCase):
  """Integration tests using proprietary models."""

  # Injection seam: filled by the --models-dir option in main(), or by a
  # subclass
  models_dir = None

  @classmethod
  def setUpClass(cls):
    if not cls.models_dir:
      raise unittest.SkipTest(
          "No models directory provided. "
          "Usage: bazel test ... --test_arg=--models-dir=/path/to/models")

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

    specs = self._load_compression_spec(model_path)
    tolerance = self._load_tolerance(model_path)
    compressed = compress.compress(flatbuffer, specs)
    verify.assert_outputs_match(flatbuffer, compressed, tolerance=tolerance)

  def _load_compression_spec(self, model_path):
    """Load compression spec from sidecar YAML file.

    Raises:
      FileNotFoundError: If no spec file is found.
    """
    spec_path = model_path.removesuffix('.tflite') + '.spec.yaml'
    if os.path.exists(spec_path):
      with open(spec_path) as f:
        return spec.parse_yaml(f.read())

    raise FileNotFoundError(
        f"No compression spec file found for {model_path}. "
        f"Expected: {spec_path}")

  def _load_tolerance(self, model_path):
    """Load tolerance from sidecar config if present.

    Returns None, meaning exact match, if no config file exists.
    """
    config_path = model_path.removesuffix('.tflite') + '.config.yaml'
    if not os.path.exists(config_path):
      return None
    with open(config_path) as f:
      config = yaml.safe_load(f)
    # float() rescues exponents like 1e-6, which YAML reads as strings
    return verify.Tolerance(rtol=float(config.get('rtol', 0)),
                            atol=float(config.get('atol', 0)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--models-dir",
      help="directory of .tflite models with sidecar compression specs")
  args, rest = parser.parse_known_args()
  if args.models_dir:
    if not os.path.isdir(args.models_dir):
      parser.error(f"not a directory: {args.models_dir}")
    ProprietaryModelTest.models_dir = args.models_dir
  unittest.main(argv=[sys.argv[0]] + rest)
