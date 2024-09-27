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

import csv

import numpy as np
import tensorflow as tf

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb


class TestDataGenerator:
  """ Generate test input/output for given model(s). A list of model(s) are taken as input.
      The generated input and output files are in csv format and created in given output folder. """

  def __init__(self, output_dir, model_paths, inputs):
    self.output_dir = output_dir
    self.model_paths = model_paths
    self.csv_filenames = []
    self.inputs = inputs
    self.input_types = {}
    self.cc_srcs = []
    self.cc_hdrs = []
    self.includes = []

  def _generate_inputs_single(self, interpreter, dtype):
    input_tensor = interpreter.tensor(
        interpreter.get_input_details()[0]['index'])
    return [
        np.random.randint(low=np.iinfo(dtype).min,
                          high=np.iinfo(dtype).max,
                          dtype=dtype,
                          size=input_tensor().shape),
    ]

  def _generate_inputs_add_sub(self, interpreter, dtype):
    input_tensor0 = interpreter.tensor(
        interpreter.get_input_details()[0]['index'])
    input_tensor1 = interpreter.tensor(
        interpreter.get_input_details()[1]['index'])
    return [
        np.random.randint(low=np.iinfo(dtype).min,
                          high=np.iinfo(dtype).max,
                          dtype=dtype,
                          size=input_tensor0().shape),
        np.random.randint(low=np.iinfo(dtype).min,
                          high=np.iinfo(dtype).max,
                          dtype=dtype,
                          size=input_tensor1().shape)
    ]

  def _generate_inputs_transpose_conv(self, interpreter, dtype):
    input_tensor0 = interpreter.tensor(0)
    filter_tensor = interpreter.tensor(1)
    input_tensor1 = interpreter.tensor(2)

    output_shape = interpreter.get_output_details()[0]['shape_signature']
    output_height = output_shape[1]
    output_width = output_shape[2]

    output_shape = np.array(
        [1, output_height, output_width,
         filter_tensor().shape[0]],
        dtype=np.int32)
    if dtype == float or dtype == np.float32 or dtype == np.float64:
      random = np.random.uniform(low=1, high=100, size=input_tensor1().shape)
      return [output_shape, random.astype(np.float32)]
    else:
      return [
          output_shape,
          np.random.randint(low=np.iinfo(dtype).min,
                            high=np.iinfo(dtype).max,
                            dtype=dtype,
                            size=input_tensor1().shape)
      ]

  def _GetTypeStringFromTensor(self, tensor):
    if tensor.dtype == np.int8:
      return 'int8'
    if tensor.dtype == np.int16:
      return 'int16'
    if tensor.dtype == np.int32:
      return 'int32'
    if tensor.dtype == float or tensor.dtype == np.float32:
      return 'float'

  def generate_golden_single_in_single_out(self):
    """ Takes a single model as input. It is expecting a list with one model.
        It then generates input and output in CSV format for that model. """

    if (len(self.model_paths) != 1):
      raise RuntimeError(f'Single model expected')
    model_path = self.model_paths[0]
    interpreter = tf.lite.Interpreter(model_path=model_path,
                                      experimental_op_resolver_type=\
                                      tf.lite.experimental.OpResolverType.BUILTIN_REF)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    if len(input_details) > 1:
      raise RuntimeError(f'Only models with one input supported')
    input_tensor = interpreter.tensor(
        interpreter.get_input_details()[0]['index'])
    output_tensor = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])

    input_type = interpreter.get_input_details()[0]['dtype']
    output_type = interpreter.get_output_details()[0]['dtype']
    if input_type != np.int8 or output_type != np.int8:
      raise RuntimeError(f'Only int8 models supported')

    generated_inputs = self._generate_inputs_single(interpreter,
                                                    input_tensor().dtype)
    for i, _input_detail in enumerate(input_details):
      interpreter.set_tensor(input_details[i]["index"], generated_inputs[i])

    interpreter.invoke()

    self._write_golden(generated_inputs, model_path, output_tensor)

  def generate_goldens(self, builtin_operator):
    """ Takes a list of one or more models as input.
        It also takes a built in operator as input because the generated input depends
        on what type of operator it is, and it supports a limited number of operators.
        All models in the list assumes the operator as the first operator. It generates
        input and output in CSV format for the corresponding models. """

    for model_path in self.model_paths:
      # Load model and run a single inference with random inputs.
      interpreter = tf.lite.Interpreter(
          model_path=model_path,
          experimental_op_resolver_type=\
          tf.lite.experimental.OpResolverType.BUILTIN_REF)
      interpreter.allocate_tensors()
      input_tensor = interpreter.tensor(
          interpreter.get_input_details()[0]['index'])
      output_tensor = interpreter.tensor(
          interpreter.get_output_details()[0]['index'])

      if builtin_operator in (schema_fb.BuiltinOperator.CONV_2D,
                              schema_fb.BuiltinOperator.DEPTHWISE_CONV_2D,
                              schema_fb.BuiltinOperator.STRIDED_SLICE,
                              schema_fb.BuiltinOperator.PAD,
                              schema_fb.BuiltinOperator.LEAKY_RELU):
        generated_inputs = self._generate_inputs_single(
            interpreter,
            input_tensor().dtype)
      elif builtin_operator in (schema_fb.BuiltinOperator.ADD,
                                schema_fb.BuiltinOperator.SUB):
        generated_inputs = self._generate_inputs_add_sub(
            interpreter,
            input_tensor().dtype)
      elif builtin_operator == schema_fb.BuiltinOperator.TRANSPOSE_CONV:
        input_tensor = interpreter.tensor(
            interpreter.get_input_details()[1]['index'])
        generated_inputs = self._generate_inputs_transpose_conv(
            interpreter,
            input_tensor().dtype)
      else:
        raise RuntimeError(f'Unsupported BuiltinOperator: {builtin_operator}')

      for idx, input_tensor_idx in enumerate(self.inputs):
        interpreter.set_tensor(input_tensor_idx, generated_inputs[idx])
      interpreter.invoke()

      self._write_golden(generated_inputs, model_path, output_tensor)

  def _write_golden(self, generated_inputs, model_path, output_tensor):
    """ Generates input and ouputs in CSV format for given model. """

    # Write input to CSV file.
    for input_idx, input_tensor_data in enumerate(generated_inputs):
      input_type = self._GetTypeStringFromTensor(input_tensor_data)
      self.input_types[input_idx] = input_type
      input_flat = input_tensor_data.flatten().tolist()
      csv_input_filename = \
          f"{model_path.split('.')[0]}_input{input_idx}_{input_type}.csv"
      input_csvfile = open(csv_input_filename, 'w', newline='')
      input_csvwriter = csv.writer(input_csvfile)
      input_csvwriter.writerow(input_flat)
      self.csv_filenames.append(csv_input_filename)

    output_flat = output_tensor().flatten().tolist()

    # Write golden to CSV file.
    output_type = self._GetTypeStringFromTensor(output_tensor())
    self.output_type = output_type
    csv_golden_filename = f"{model_path.split('.')[0]}_golden_{output_type}.csv"
    golden_csvfile = open(csv_golden_filename, 'w', newline='')
    golden_csvwriter = csv.writer(golden_csvfile)
    np.set_printoptions(threshold=np.inf)
    golden_csvwriter.writerow(output_flat)
    self.csv_filenames.append(csv_golden_filename)

  def generate_makefile(self,
                        test_file='integration_tests.cc',
                        src_prefix=None):
    """ Generates a makefile which takes the the given input model(s) as input and also the
        corresponding generated input(s) and ouput(s) in csv format. It also take the name of a test file as input.
        For example usage see: tensorflow/lite/micro/integration_tests/generate_per_layer_tests.py. """

    makefile = open(self.output_dir + '/Makefile.inc', 'w')
    output_dir_list = self.output_dir.split('/')
    if src_prefix is None:
      src_prefix = output_dir_list[-3] + '_' + output_dir_list[
          -2] + '_' + output_dir_list[-1]
    makefile.write(src_prefix + '_GENERATOR_INPUTS := \\\n')
    for model_path in self.model_paths:
      makefile.write('$(TENSORFLOW_ROOT)' +
                     model_path.split('third_party/tflite_micro/')[-1] +
                     ' \\\n')
    for csv_input in self.csv_filenames:
      makefile.write('$(TENSORFLOW_ROOT)' +
                     csv_input.split('third_party/tflite_micro/')[-1] +
                     ' \\\n')
    makefile.write('\n')
    makefile.write(src_prefix + '_SRCS := \\\n')
    makefile.write('$(TENSORFLOW_ROOT)' +
                   self.output_dir.split('third_party/tflite_micro/')[-1] +
                   '/' + test_file + '  \\\n')
    makefile.write(
        "$(TENSORFLOW_ROOT)python/tflite_micro/python_ops_resolver.cc \\\n")
    makefile.write('\n\n')
    makefile.write(src_prefix + '_HDR := \\\n')
    makefile.write(
        "$(TENSORFLOW_ROOT)python/tflite_micro/python_ops_resolver.h \\\n")
    makefile.write('\n\n')
    makefile.write('$(eval $(call microlite_test,' + src_prefix + '_test,\\\n')
    makefile.write('$(' + src_prefix + '_SRCS),$(' + src_prefix + '_HDR),$(' +
                   src_prefix + '_GENERATOR_INPUTS)))')
