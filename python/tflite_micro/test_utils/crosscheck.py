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
"""Crosscheck the Micro interpreter against other interpreters.

Crosscheck the Micro interpreter's output against a reference interpreter's
output. The passed model and a randomly generated input tensor are fed both
interpreters, and the outputs are compared for equality. All outputs, inputs,
and the state of both interpreters are returned for comparison and debugging.

Several types of models are accepted:

  - in-memory tflite flatbuffer
  - tflite flatbuffer file
  - Keras model

Keras models are naïvely converted to a quantized tflite flatbuffer by the
tf.lite.TFLiteConverter, using a randomly generated representative input. For
precise control over the conversion, convert the model prior to submission.

Current limitations:

  - the only reference interpreter implemented is Lite
  - assumes int8 inputs and outputs

Typical usage:

  result = crosscheck.versus_lite(tflite_model=model_data)

  if result:
    # a true result indicates the outputs match
      pass
    else:
      # debug a failure
      print(result)  # input, output, and diff convert to a string
      print(result.micro.X))  # interpreter state is available for inspection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


class Result:
  """The result of a crosscheck between Micro and a reference.

  The result of a crosscheck between the Micro interpreter and a reference
  interpreter. Normally this is returned by a crosscheck.versus_* function, and
  not directly constructed by users of the crosscheck module.

  A Result evaluates to True in a boolean context if the output of the two
  interpreters matches; and an informal string representation containing the
  input, outputs, and diff when used by the built-in str(), format(), and
  print() functions. Additionally, several attributes and methods are provided
  for debugging the crosscheck run.

  Attributes:
    input: The input tensor fed to both interpreters.
    micro: The Micro interpreter, in its post-invoke state.
    reference: The reference interpreter, in its post-invoke state.
  """

  def __init__(self, input, micro, reference, rtol=None, atol=None):
    """Constructor.

    Not normally called by users of the module, but by crosscheck.versus_*
    functions to return their results.

    Args:
      input:
        The input tensor fed to both interpreters.
      micro:
        The Micro interpreter, in its post-invoke state.
      reference:
        The reference interpreter, in its post-invoke state.
      rtol:
        The relative tolerance to use when comparing the outputs. See
        np.allclose(). If None, defaults to that of np.allclose(). May be
        overridden in the outputs_match() method.
      atol:
        The absolute tolerance to use when comparing the outputs. See
        np.allclose(). If None, defaults to that of np.allclose(). May be
        overridden in the outputs_match() method.
    """
    self.input = input
    self.micro = micro
    self.reference = reference

    self._rtol = rtol
    self._atol = atol

  def micro_output(self):
    """Get the output tensor of the Micro interpreter."""
    return self.micro.get_output(0)

  def reference_output(self):
    """Get the output tensor of the reference interpreter."""
    index = self.reference.get_output_details()[0]["index"]
    return self.reference.get_tensor(index)

  def output_diff(self):
    """Get a tensor with the difference of the outputs.

    Get a tensor that is the Micro output subtracted from the reference output.
    Were the outputs match, the difference should be zero or nearly zero.
    """
    return (tf.convert_to_tensor(self.micro_output(), dtype=tf.int16) -
            tf.convert_to_tensor(self.reference_output(), dtype=tf.int16))

  def outputs_match(self, rtol=None, atol=None):
    """True if outputs are element-wise equal.

    Returns True if outputs are element-wise equal according to np.allclose().
    Optional tolerances override the defaults with which the Result was
    constructed.

    Args:
      rtol: Optional relative tolerance, see np.allclose().
      atol: Optional absolute tolerance, see np.allclose().
    """

    def insert_if_not_none(dict, key, value):
      if value is not None:
        dict[key] = value

    kwargs = {}
    insert_if_not_none(kwargs, "rtol", self._rtol)
    insert_if_not_none(kwargs, "rtol", rtol)
    insert_if_not_none(kwargs, "atol", self._atol)
    insert_if_not_none(kwargs, "atol", atol)

    return np.allclose(self.micro_output(), self.reference_output(), **kwargs)

  def __bool__(self):
    """True if the outputs match."""

    return self.outputs_match()

  def __str__(self):
    """Return a string containing input, outputs, and diff."""

    return (f"input: shape={self.input.shape}\n"
            f"{self.input}\n"
            f"\n"
            f"lite_output: shape={self.reference_output().shape}\n"
            f"{tf.convert_to_tensor(self.reference_output())}\n"
            f"\n"
            f"micro_output: shape={self.micro_output().shape}\n"
            f"{tf.convert_to_tensor(self.micro_output())}\n"
            f"\n"
            f"diff: shape={self.output_diff().shape}\n"
            f"{self.output_diff()}")


def _make_int8_tflite_model(keras_model, rng):
  """Convert a Keras model to an int8 tflite model.

  Convert the Keras model to an int8 tflite model using
  tf.lite.TFLiteConverter. Quantization is done using randomly generated
  input data. The generator is taken as a parameter so the caller can control
  seeding and or other properties of the input.

  Args:
    keras_model:
      The model to convert.
    rng:
      An np.random.Generator used to generate the input data for
      quantization.
  """

  def random_representative_dataset():
    shape = keras_model.layers[0].input_shape[0][1:]
    arbitrary_length = 100
    for _ in range(arbitrary_length):
      data = rng.random(size=(1, *shape), dtype=np.float32)
      yield [data]

  converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
  converter.representative_dataset = random_representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  return tflite_model


def versus_lite(*,
                keras_model=None,
                tflite_model=None,
                tflite_path=None,
                tflm_interpreter=tflm_runtime.Interpreter,
                rng=np.random.default_rng(seed=42),
                rtol=None,
                atol=None):
  """Check the Micro interpreter against the Lite interpreter.

  Check the output of the Micro interpreter using the output of the
  tf.Lite.Interpreter as a reference, as described in the module docstring.

  Pass one and only one of the args keras_module, tflite_model, and
  tflite_path.

  Args:
    keras_model:
      A Keras model.
    tflite_model:
      Model data in tflite format
    tflite_path:
      A model file in tflite format
    tflm_interpreter:
      The Micro Interpreter class to instantiate. This is parametrizable to
      allow customization (for testing, among other possibilities) by
      inheriting from the main Micro Interpreter.
    rng:
      An np.random.Generator used to generate the input data for testing and
      quantization of models. This is parametrizable to permit control of
      seeding and or other properties of the input.
    rtol:
      The relative tolerance used to compare the outputs, see np.allclose().
      Defaults to that of np.allclose().
    atol:
      The absolute tolerance used to compare the outputs, see np.allclose().
      Defaults to that of np.allclose().

  Returns:
      A Result object.
  """
  if keras_model is not None:
    model = _make_int8_tflite_model(keras_model, rng)
  elif tflite_model is not None:
    model = tflite_model
  elif tflite_path is not None:
    with open(tflite_path, mode='rb') as f:
      model = f.read()

  lite = tf.lite.Interpreter(model_content=model)
  lite.allocate_tensors()
  input_shape = lite.get_input_details()[0]["shape"][1:]

  input = rng.integers(-128, 127, size=(1, *input_shape), dtype=np.int8)

  lite.set_tensor(lite.get_input_details()[0]["index"], input)
  lite.invoke()

  micro = tflm_interpreter.from_bytes(model)
  micro.set_input(input, 0)
  micro.invoke()

  return Result(input, micro, lite, rtol=rtol, atol=atol)
