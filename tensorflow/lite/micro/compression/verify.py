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
"""Verify that two models produce matching outputs.

Feed identical random inputs to two models through the TFLM Python
interpreter and compare their outputs. The primary use is checking that
a compressed model behaves the same as the original it was compressed
from.
"""

import typing

import numpy as np

from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import tensor_type

_TRIALS = 5
_SEED = 42


def assert_outputs_match(original, candidate, *, tolerance=None):
  """Assert that two models produce matching outputs on random inputs.

  Run both models on identical random inputs for several trials.
  Outputs must match exactly, unless a Tolerance is given, in which
  case an output element passes when:
      abs(candidate - original) <= atol + rtol * abs(original)

  Raises:
    AssertionError: if any output mismatches.
  """
  original_interp = runtime.Interpreter.from_bytes(bytes(original))
  candidate_interp = runtime.Interpreter.from_bytes(bytes(candidate))

  rng = np.random.default_rng(_SEED)
  sg = model_editor.read(original).subgraphs[0]

  for trial in range(_TRIALS):
    for i, tensor in enumerate(sg.inputs):
      value = _random_input(rng, tensor)
      original_interp.set_input(value, i)
      candidate_interp.set_input(value, i)

    original_interp.invoke()
    candidate_interp.invoke()

    for i in range(len(sg.outputs)):
      expected = original_interp.get_output(i)
      actual = candidate_interp.get_output(i)
      msg = f"Output mismatch (trial {trial}, output {i})"
      if tolerance is None:
        np.testing.assert_array_equal(expected, actual, err_msg=msg)
      else:
        np.testing.assert_allclose(expected,
                                   actual,
                                   rtol=tolerance.rtol,
                                   atol=tolerance.atol,
                                   err_msg=msg)


class Tolerance(typing.NamedTuple):
  """Output comparison tolerances for assert_outputs_match."""
  rtol: float
  atol: float


def _random_input(rng, tensor):
  """Generate a random array matching the tensor's shape and dtype."""
  shape = tensor.shape
  dtype = tensor_type.to_numpy(tensor.dtype)

  if np.issubdtype(dtype, np.floating):
    return rng.uniform(-1.0, 1.0, shape).astype(dtype)
  info = np.iinfo(dtype)
  return rng.integers(info.min, info.max, shape, dtype=dtype, endpoint=True)
