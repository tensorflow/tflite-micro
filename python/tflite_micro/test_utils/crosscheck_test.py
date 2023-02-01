# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#
"""Unit tests for crosscheck"""

import numpy as np
from tensorflow import keras
from tensorflow import test
from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.test_utils import crosscheck
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime

_TEST_MODEL_PATH = resource_loader.get_path_to_datafile(
    "crosscheck_test.model.tflite")


class InputTest(test.TestCase):
  """Test with each supported type of model input."""

  def test_tflite_path(self):
    result = crosscheck.versus_lite(tflite_path=_TEST_MODEL_PATH)
    self.assertTrue(result, msg=result)

  def test_tflite_model(self):
    with open(_TEST_MODEL_PATH, 'rb') as f:
      model = f.read()

    result = crosscheck.versus_lite(tflite_model=model)
    self.assertTrue(result, msg=result)

  def test_keras_model(self):
    arbitrary_shape = (16, 1, 3)
    first = keras.Input(shape=arbitrary_shape, batch_size=1)
    last = keras.layers.Dense(arbitrary_shape[0], activation="relu")(first)
    model = keras.Model(first, last)
    model.compile()

    result = crosscheck.versus_lite(keras_model=model)
    self.assertTrue(result, msg=result)


class _OffsetInterpreter(tflm_runtime.Interpreter):
  """A Micro interpreter fake that produces incorrect output.

  Wrap the Micro interpreter, intentionally producing incorrect output by
  adding OFFSET to each element. Saturate at the maximum value of the output
  data type, so that elements remain within OFFSET of the correct value rather
  than overflowing and wrapping around. This permits testing for equality
  within a tolerance.
  """

  OFFSET = 5

  def get_output(self, index):
    # Unspoiled output from real interpreter
    output = super().get_output(index)

    # Add offset, but saturate at the max value of the dtype
    max_value = np.iinfo(output.dtype).max
    add_with_saturation = np.vectorize(
        lambda x: min(x + _OffsetInterpreter.OFFSET, max_value))

    return add_with_saturation(output)


class FailureTest(test.TestCase):

  def test_with_offset(self):
    result = crosscheck.versus_lite(tflite_path=_TEST_MODEL_PATH,
                                    tflm_interpreter=_OffsetInterpreter)
    self.assertFalse(result, msg=result)


class ToleranceTest(test.TestCase):

  def test_outside_tolerance(self):
    result = crosscheck.versus_lite(tflite_path=_TEST_MODEL_PATH,
                                    tflm_interpreter=_OffsetInterpreter,
                                    atol=_OffsetInterpreter.OFFSET)
    self.assertTrue(result, msg=result)


if __name__ == "__main__":
  test.main()
