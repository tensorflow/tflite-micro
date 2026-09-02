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
"""Smoke test for the proprietary-model integration test harness.

Run the harness in proprietary_integration_test.py over a temporary
models directory holding synthetic models, so the harness's model
discovery, sidecar parsing, input generation, and output comparison
run in normal CI, where no proprietary model is available.
"""

import pathlib
import tempfile
import unittest

from tflite_micro.tensorflow.lite.micro.compression import compression_integration_test
from tflite_micro.tensorflow.lite.micro.compression import proprietary_integration_test

_SPEC_YAML = """\
tensors:
  - subgraph: 0
    tensor: 0
    compression:
      - lut:
          index_bitwidth: 2
"""


class HarnessSmokeTest(proprietary_integration_test.ProprietaryModelTest):
  """The harness's own tests, run over a generated models directory.

  One model pairs with only a spec file and takes the exact-match
  path; the other adds a config file and takes the tolerance path.
  """

  @classmethod
  def setUpClass(cls):
    cls._tmpdir = tempfile.TemporaryDirectory()
    d = pathlib.Path(cls._tmpdir.name)
    flatbuffer = compression_integration_test._build_compressible_model()
    (d / "exact.tflite").write_bytes(flatbuffer)
    (d / "exact.spec.yaml").write_text(_SPEC_YAML)
    (d / "tolerant.tflite").write_bytes(flatbuffer)
    (d / "tolerant.spec.yaml").write_text(_SPEC_YAML)
    (d / "tolerant.config.yaml").write_text("rtol: 1.0e-6\natol: 1.0e-6\n")
    cls.models_dir = str(d)
    super().setUpClass()

  @classmethod
  def tearDownClass(cls):
    cls._tmpdir.cleanup()


if __name__ == "__main__":
  unittest.main()
