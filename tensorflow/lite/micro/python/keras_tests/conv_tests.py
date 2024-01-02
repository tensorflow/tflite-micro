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
# =============================================================================
"""
Convolution kernel testing with dilation > 1, using the TfLiteConverter to
convert models directly from Keras.

Run:
bazel build tensorflow/lite/micro/python/keras_tests:conv_tests
bazel-bin/tensorflow/lite/micro/python/keras_tests/conv_tests
"""

from __future__ import annotations

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

import tensorflow as tf
import keras.api._v2.keras as keras  # for Visual Studio Code to work correctly
from tflite_micro.python.tflite_micro import runtime


class KerasConvTest(test_util.TensorFlowTestCase):

  def MakeConv1dModel(self, *, shape, dilation):
    input_layer = keras.layers.Input(shape=shape)
    conv_layer = keras.layers.Conv1D(1,
                                     3,
                                     dilation_rate=dilation,
                                     padding='same')(input_layer)
    model = keras.Model(inputs=input_layer, outputs=conv_layer)
    return model

  def MakeConv2dModel(self, *, shape, dilation):
    input_layer = keras.layers.Input(shape=shape)
    conv_layer = keras.layers.Conv2D(1,
                                     3,
                                     dilation_rate=dilation,
                                     padding='same')(input_layer)
    model = keras.Model(inputs=input_layer, outputs=conv_layer)
    return model

  def MakeDepthwiseConv1dModel(self, *, shape, dilation):
    input_layer = keras.layers.Input(shape=shape)
    conv_layer = keras.layers.DepthwiseConv1D(3,
                                              dilation_rate=dilation,
                                              padding='same')(input_layer)
    model = keras.Model(inputs=input_layer, outputs=conv_layer)
    return model

  def MakeDepthwiseConv2dModel(self, *, shape, dilation):
    input_layer = keras.layers.Input(shape=shape)
    conv_layer = keras.layers.DepthwiseConv2D(3,
                                              dilation_rate=dilation,
                                              padding='same')(input_layer)
    model = keras.Model(inputs=input_layer, outputs=conv_layer)
    return model

  def MakeTransposeConv1dModel(self, *, shape, dilation):
    input_layer = keras.layers.Input(shape=shape)
    conv_layer = keras.layers.Conv1DTranspose(1,
                                              3,
                                              dilation_rate=dilation,
                                              padding='same')(input_layer)
    model = keras.Model(inputs=input_layer, outputs=conv_layer)
    return model

  def MakeTransposeConv2dModel(self, *, shape, dilation):
    input_layer = keras.layers.Input(shape=shape)
    conv_layer = keras.layers.Conv2DTranspose(1,
                                              3,
                                              dilation_rate=dilation,
                                              padding='same')(input_layer)
    model = keras.Model(inputs=input_layer, outputs=conv_layer)
    return model

  def ExecuteModelTest(self, model: keras.Model):
    model_shape = list(model.layers[0].input_shape[0])
    model_shape[0] = 1
    input_data = tf.ones(shape=model_shape, dtype=tf.float32)
    tf_result: tf.Tensor = model(input_data)  # type: ignore

    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

    tflm_interpreter = runtime.Interpreter.from_bytes(
        tflite_model,
        intrepreter_config=runtime.InterpreterConfig.kPreserveAllTensors)
    tflm_interpreter.set_input(input_data, 0)
    tflm_interpreter.invoke()
    tflm_result = tflm_interpreter.get_output(0)
    tflm_output_details = tflm_interpreter.get_output_details(0)
    tflm_shape = tflm_output_details['shape']

    print(f'{tf_result=}')
    print(f'{tflm_result=} {tflm_shape=}')

    self.assertAllClose(tf_result, tflm_result)
    self.assertAllEqual(tf_result.shape, tflm_shape)

  def setUp(self):
    pass

  def testConv1dWithDilation1(self):
    model = self.MakeConv1dModel(shape=(8, 1), dilation=1)
    self.ExecuteModelTest(model)

  def testConv1dWithDilation2(self):
    model = self.MakeConv1dModel(shape=(8, 1), dilation=2)
    self.ExecuteModelTest(model)

  def testConv2dWithDilation1(self):
    model = self.MakeConv2dModel(shape=(1, 8, 1), dilation=1)
    self.ExecuteModelTest(model)

  def testConv2dWithDilation2(self):
    model = self.MakeConv2dModel(shape=(1, 8, 1), dilation=2)
    self.ExecuteModelTest(model)

  def testDepthwiseConv1dWithDilation1(self):
    model = self.MakeDepthwiseConv1dModel(shape=(8, 1), dilation=1)
    self.ExecuteModelTest(model)

  def testDepthwiseConv1dWithDilation2(self):
    model = self.MakeDepthwiseConv1dModel(shape=(8, 1), dilation=2)
    self.ExecuteModelTest(model)

  def testDepthwiseConv2dWithDilation1(self):
    model = self.MakeDepthwiseConv2dModel(shape=(1, 8, 1), dilation=1)
    self.ExecuteModelTest(model)

  def testDepthwiseConv2dWithDilation2(self):
    model = self.MakeDepthwiseConv2dModel(shape=(1, 8, 1), dilation=2)
    self.ExecuteModelTest(model)

  def testTransposeConv1dWithDilation1(self):
    model = self.MakeTransposeConv1dModel(shape=(8, 1), dilation=1)
    self.ExecuteModelTest(model)

  def testTransposeConv2dWithDilation1(self):
    model = self.MakeTransposeConv2dModel(shape=(1, 8, 1), dilation=1)
    self.ExecuteModelTest(model)


if __name__ == '__main__':
  test.main()
