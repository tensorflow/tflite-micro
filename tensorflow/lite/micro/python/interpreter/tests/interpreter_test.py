# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Basic Python test for the TFLM interpreter"""

# Steps to run this test:
#   bazel test tensorflow/lite/micro/python/interpreter/tests:interpreter_test
#
# Steps to debug with gdb:
# 1. bazel build tensorflow/lite/micro/python/interpreter/tests:interpreter_test
# 2. gdb python
# 3. (gdb) run bazel-out/k8-fastbuild/bin/tensorflow/lite/micro/python/interpreter/tests/interpreter_test

import gc
import weakref
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.testing import generate_test_models
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


class ConvModelTests(test_util.TensorFlowTestCase):
  filename = "/tmp/interpreter_test_conv_model.tflite"
  input_shape = (1, 16, 16, 1)
  output_shape = (1, 10)

  def testInitErrorHandling(self):
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             "Invalid model file path"):
      tflm_runtime.Interpreter.from_file("wrong.tflite")

  def testInput(self):
    model_data = generate_test_models.generate_conv_model(False)
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)

    data_x = np.random.randint(-127, 127, self.input_shape, dtype=np.int8)
    tflm_interpreter.set_input(data_x, 0)

    # Test input tensor details
    input_details = tflm_interpreter.get_input_details(0)
    self.assertAllEqual(input_details["shape"], self.input_shape)
    # Single channel int8 quantization
    self.assertEqual(input_details["dtype"], np.int8)
    self.assertEqual(len(input_details["quantization_parameters"]["scales"]),
                     1)
    self.assertEqual(
        input_details["quantization_parameters"]["quantized_dimension"], 0)
    # TODO(b/247808903): check only the types here to make sure that all arrays are properly set up.
    self.assertEqual(input_details["quantization_parameters"]["scales"].dtype,
                     np.float32)
    self.assertEqual(
        input_details["quantization_parameters"]["zero_points"].dtype,
        np.int32)

  def testInputErrorHandling(self):
    model_data = generate_test_models.generate_conv_model(True, self.filename)
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)

    data_x = np.random.randint(-127, 127, self.input_shape, dtype=np.int8)
    # Try to access out of bound data
    with self.assertRaisesWithPredicateMatch(IndexError,
                                             "Tensor is out of bound"):
      tflm_interpreter.set_input(data_x, 1)
    # Pass data with wrong dimension
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             "Dimension mismatch."):
      reshaped_data = data_x.reshape((1, 16, 16, 1, 1))
      tflm_interpreter.set_input(reshaped_data, 0)
    # Pass data with wrong dimension in one axis
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             "Dimension mismatch."):
      reshaped_data = data_x.reshape((1, 2, 128, 1))
      tflm_interpreter.set_input(reshaped_data, 0)
    # Pass data with wrong type
    with self.assertRaisesWithPredicateMatch(ValueError, "Got value of type"):
      float_data = data_x.astype(np.float32)
      tflm_interpreter.set_input(float_data, 0)
    # Reach wrong details
    with self.assertRaisesWithPredicateMatch(IndexError,
                                             "Tensor is out of bound"):
      tflm_interpreter.get_input_details(1)

  def testOutput(self):
    model_data = generate_test_models.generate_conv_model(True, self.filename)
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)

    # Initial output values are all 0
    output = tflm_interpreter.get_output(0)
    init_output = np.zeros(self.output_shape)
    self.assertAllEqual(output, init_output)

    # Test the output tensor details
    output_details = tflm_interpreter.get_output_details(0)
    self.assertAllEqual(output_details["shape"], self.output_shape)
    # Single channel int8 quantization
    self.assertEqual(output_details["dtype"], np.int8)
    self.assertEqual(len(output_details["quantization_parameters"]["scales"]),
                     1)
    self.assertEqual(
        output_details["quantization_parameters"]["quantized_dimension"], 0)
    # TODO(b/247808903): check only the types here to make sure that all arrays are properly set up.
    self.assertEqual(output_details["quantization_parameters"]["scales"].dtype,
                     np.float32)
    self.assertEqual(
        output_details["quantization_parameters"]["zero_points"].dtype,
        np.int32)

  def testOutputErrorHandling(self):
    model_data = generate_test_models.generate_conv_model(True, self.filename)
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)
    # Try to access out of bound data
    with self.assertRaisesWithPredicateMatch(IndexError,
                                             "Tensor is out of bound"):
      tflm_interpreter.get_output(1)
    with self.assertRaisesWithPredicateMatch(IndexError,
                                             "Tensor is out of bound"):
      tflm_interpreter.get_output_details(1)

  def testCompareWithTFLite(self):
    model_data = generate_test_models.generate_conv_model(True, self.filename)

    # TFLM interpreter
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)

    # TFLite interpreter
    tflite_interpreter = tf.lite.Interpreter(
        model_content=model_data,
        experimental_op_resolver_type=\
        tf.lite.experimental.OpResolverType.BUILTIN_REF)
    tflite_interpreter.allocate_tensors()
    tflite_output_details = tflite_interpreter.get_output_details()[0]
    tflite_input_details = tflite_interpreter.get_input_details()[0]

    num_steps = 100
    for i in range(0, num_steps):
      # Create random input
      data_x = np.random.randint(-127, 127, self.input_shape, dtype=np.int8)

      # Run inference on TFLite
      tflite_interpreter.set_tensor(tflite_input_details["index"], data_x)
      tflite_interpreter.invoke()
      tflite_output = tflite_interpreter.get_tensor(
          tflite_output_details["index"])

      # Run inference on TFLM
      tflm_interpreter.set_input(data_x, 0)
      tflm_interpreter.invoke()
      tflm_output = tflm_interpreter.get_output(0)

      # Check that TFLM output has correct metadata
      self.assertDTypeEqual(tflm_output, np.int8)
      self.assertEqual(tflm_output.shape, self.output_shape)
      self.assertAllEqual(tflite_output, tflm_output)

  def _helperModelFromFileAndBufferEqual(self):
    model_data = generate_test_models.generate_conv_model(True, self.filename)

    file_interpreter = tflm_runtime.Interpreter.from_file(self.filename)
    bytes_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)

    num_steps = 100
    for i in range(0, num_steps):
      data_x = np.random.randint(-127, 127, self.input_shape, dtype=np.int8)

      file_interpreter.set_input(data_x, 0)
      file_interpreter.invoke()
      file_output = file_interpreter.get_output(0)

      bytes_interpreter.set_input(data_x, 0)
      bytes_interpreter.invoke()
      bytes_output = bytes_interpreter.get_output(0)

      self.assertDTypeEqual(file_output, np.int8)
      self.assertEqual(file_output.shape, self.output_shape)
      self.assertDTypeEqual(bytes_output, np.int8)
      self.assertEqual(bytes_output.shape, self.output_shape)
      # Same interpreter and model, should expect all equal
      self.assertAllEqual(file_output, bytes_output)

  def testModelFromFileAndBufferEqual(self):
    self._helperModelFromFileAndBufferEqual()

  def testMultipleInterpreters(self):
    model_data = generate_test_models.generate_conv_model(False)

    interpreters = [
        tflm_runtime.Interpreter.from_bytes(model_data) for i in range(10)
    ]

    num_steps = 100
    for i in range(0, num_steps):
      data_x = np.random.randint(-127, 127, self.input_shape, dtype=np.int8)

      prev_output = None
      for interpreter in interpreters:
        interpreter.set_input(data_x, 0)
        interpreter.invoke()
        output = interpreter.get_output(0)
        if prev_output is None:
          prev_output = output

        self.assertDTypeEqual(output, np.int8)
        self.assertEqual(output.shape, self.output_shape)
        self.assertAllEqual(output, prev_output)

  def _helperNoop(self):
    pass

  def _helperOutputTensorMemoryLeak(self):
    interpreter = tflm_runtime.Interpreter.from_file(self.filename)
    int_ref = weakref.finalize(interpreter, self._helperNoop)
    some_output = interpreter.get_output(0)
    output_ref = weakref.finalize(some_output, self._helperNoop)
    return (int_ref, output_ref)

  def testOutputTensorMemoryLeak(self):
    generate_test_models.generate_conv_model(True, self.filename)

    int_ref, output_ref = self._helperOutputTensorMemoryLeak()
    # Output obtained in the helper function should be out of scope now, perform
    # garbage collection and check that the weakref is dead. If it's still
    # alive, it means that the output's reference count isn't 0 by garbage
    # collection. Since it's already out of scope, this means a memory leak.
    #
    # An example of how this could be true is if there's an additional
    # reference increment (e.g. `Py_INCREF` or `py::cast`` instead of
    # `py::reinterpret_steal``) somewhere in the C++ code.
    gc.collect()
    self.assertFalse(int_ref.alive)
    self.assertFalse(output_ref.alive)

  # TODO (b/240162715): Add a test case to register a custom OP

  def testMalformedCustomOps(self):
    model_data = generate_test_models.generate_conv_model(False)
    custom_op_registerers = [("wrong", "format")]
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             "must be a list of strings"):
      interpreter = tflm_runtime.Interpreter.from_bytes(
          model_data, custom_op_registerers)

    custom_op_registerers = "WrongFormat"
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             "must be a list of strings"):
      interpreter = tflm_runtime.Interpreter.from_bytes(
          model_data, custom_op_registerers)

  def testNonExistentCustomOps(self):
    model_data = generate_test_models.generate_conv_model(False)
    custom_op_registerers = ["SomeRandomOp"]
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, "TFLM could not register custom op via SomeRandomOp"):
      interpreter = tflm_runtime.Interpreter.from_bytes(
          model_data, custom_op_registerers)


if __name__ == "__main__":
  test.main()
