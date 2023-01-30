import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import \
  tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.hello_world import evaluate

PREFIX_PATH = resource_loader.get_path_to_datafile('')


class HelloWorldQuantModelTest(test_util.TensorFlowTestCase):
  model_path = os.path.join(PREFIX_PATH, 'hello_world.tflite')
  input_shape = (1, 1)
  output_shape = (1, 1)
  # Create the tflm interpreter
  tflm_interpreter = tflm_runtime.Interpreter.from_file(model_path)

  def test_init_error_handling(self):
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             'Invalid model file path'):
      tflm_runtime.Interpreter.from_file('wrong.tflite')

  def test_input(self):
    input_details = self.tflm_interpreter.get_input_details(0)
    input_scale, input_zero_point = evaluate.get_metadata(input_details)

    self.assertAllEqual(input_details['shape'], self.input_shape)
    self.assertEqual(input_details['dtype'], np.int8)
    self.assertEqual(len(input_scale), 1)
    self.assertEqual(
        input_details['quantization_parameters']['quantized_dimension'], 0)
    self.assertEqual(input_scale.dtype, np.float32)
    self.assertEqual(input_zero_point.dtype, np.int32)

  def test_output(self):
    output_details = self.tflm_interpreter.get_output_details(0)
    output_scale, output_zero_point = evaluate.get_metadata(output_details)

    self.assertAllEqual(output_details['shape'], self.output_shape)
    self.assertEqual(output_details['dtype'], np.int8)
    self.assertEqual(len(output_scale), 1)
    self.assertEqual(
        output_details['quantization_parameters']['quantized_dimension'], 0)
    self.assertEqual(output_scale.dtype, np.float32)
    self.assertEqual(output_zero_point.dtype, np.int32)

  def test_interpreter_prediction(self):
    x_value = 0.0
    # Calculate the corresponding sine values
    y_true = np.sin(x_value).astype(np.float32)

    input_details = self.tflm_interpreter.get_input_details(0)
    input_scale, input_zero_point = evaluate.get_metadata(input_details)

    output_details = self.tflm_interpreter.get_output_details(0)
    output_scale, output_zero_point = evaluate.get_metadata(output_details)

    input_shape = np.array(input_details.get('shape'))

    x_quantized = np.int8((x_value / input_scale[0]) + input_zero_point[0])
    y_quantized = evaluate.invoke_tflm_interpreter(
        input_shape,
        self.tflm_interpreter,
        x_quantized,
        input_index=0,
        output_index=0)
    y_pred = float((y_quantized - output_zero_point[0]) * output_scale[0])
    epsilon = 0.05
    self.assertNear(
        y_true, y_pred, epsilon,
        'hello_world model prediction is not close enough to numpy.sin value')

  def test_compare_with_tflite(self):
    # TFLite interpreter
    tflite_interpreter = tf.lite.Interpreter(
        model_path=self.model_path,
        experimental_op_resolver_type= \
          tf.lite.experimental.OpResolverType.BUILTIN_REF)
    tflite_interpreter.allocate_tensors()

    x_values = evaluate.generate_random_input()

    tflm_y_predictions = evaluate.get_tflm_prediction(self.tflm_interpreter,
                                                      x_values)

    tflite_y_predictions = evaluate.get_tflite_prediction(
        tflite_interpreter, x_values)

    self.assertAllEqual(tflm_y_predictions, tflite_y_predictions)


if __name__ == '__main__':
  test.main()
