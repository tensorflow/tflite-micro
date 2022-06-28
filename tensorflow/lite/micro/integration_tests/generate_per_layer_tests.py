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

import os
import sys
import copy
import csv

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import random as rand
from mako import template

from tensorflow.python.platform import gfile
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.python import schema_util
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils
from tflite_micro.tensorflow.lite.micro.tools import generate_test_for_model

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
TEMPLATE_DIR = os.path.abspath(TEMPLATE_DIR)


def BytesFromFlatbufferType(tensor_type):
  if tensor_type in (schema_fb.TensorType.INT8, schema_fb.TensorType.UINT8,
                     schema_fb.TensorType.BOOL):
    return 1
  elif tensor_type in (schema_fb.TensorType.INT16,
                       schema_fb.TensorType.FLOAT16):
    return 2
  elif tensor_type in (schema_fb.TensorType.FLOAT32,
                       schema_fb.TensorType.INT32,
                       schema_fb.TensorType.UINT32):
    return 4
  elif tensor_type in (schema_fb.TensorType.FLOAT64,
                       schema_fb.TensorType.INT64,
                       schema_fb.TensorType.COMPLEX64,
                       schema_fb.TensorType.UINT64):
    return 8
  else:
    raise RuntimeError(f'Unsupported TensorType: {tensor_type}')


class TestModelGenerator:
  """Generates test data from tflite file."""

  def __init__(self, model, output_dir, inputs):
    self.model = model
    self.output_dir = output_dir
    self.op_idx = 0
    self.inputs = inputs

  def generate_single_layer_model(self, model, subgraph, op, opcode_idx):
    generated_model = schema_fb.ModelT()
    generated_model.buffers = []
    generated_model.version = 3

    # Create subgraph.
    generated_subgraph = schema_fb.SubGraphT()
    generated_subgraph.inputs = self.inputs
    generated_subgraph.outputs = [len(op.inputs)]
    generated_subgraph.tensors = []
    for input_idx, tensor_idx in enumerate(op.inputs):
      tensor = copy.deepcopy(subgraph.tensors[tensor_idx])
      tensor.buffer = len(generated_model.buffers)
      buffer = copy.deepcopy(
          model.buffers[subgraph.tensors[tensor_idx].buffer])
      if input_idx in self.inputs:
        buffer.data = None
      bytes_per_element = BytesFromFlatbufferType(tensor.type)
      if buffer.data is not None and len(tensor.shape) > 2:
        for i in range(len(buffer.data)):
          buffer.data[i] = buffer.data[i] * np.random.uniform(
              low=0.5, high=1.0, size=1)

        all_equal = True
        for i, elem in enumerate(buffer.data):
          all_equal = all_equal and elem == model.buffers[
              subgraph.tensors[tensor_idx].buffer].data[i]
        assert not all_equal

      generated_model.buffers.append(buffer)
      generated_subgraph.tensors.append(tensor)

    for tensor_idx in op.outputs:
      tensor = copy.deepcopy(subgraph.tensors[tensor_idx])
      tensor.buffer = len(generated_model.buffers)
      buffer = copy.deepcopy(
          model.buffers[subgraph.tensors[tensor_idx].buffer])
      generated_model.buffers.append(buffer)
      generated_subgraph.tensors.append(tensor)

    # Create op.
    generated_op = copy.deepcopy(op)
    generated_op.inputs = [i for i in range(len(op.inputs))]
    generated_op.outputs = [len(op.inputs)]
    generated_op.opcodeIndex = 0
    generated_subgraph.operators = [generated_op]

    generated_model.subgraphs = [generated_subgraph]
    generated_model.operatorCodes = [model.operatorCodes[opcode_idx]]
    model_name = self.output_dir + '/' + self.output_dir.split('/')[-1] + str(
        self.op_idx) + '.tflite'
    self.op_idx += 1
    flatbuffer_utils.write_model(generated_model, model_name)
    return model_name

  def get_opcode_idx(self, builtin_operator):
    for idx, opcode in enumerate(self.model.operatorCodes):
      if schema_util.get_builtin_code_from_operator_code(
          opcode) == builtin_operator:
        return idx

  def generate_models(self, subgraph_idx, builtin_operator):
    subgraph = self.model.subgraphs[subgraph_idx]
    opcode_idx = self.get_opcode_idx(builtin_operator)
    output_models = []
    for op in subgraph.operators:
      if op.opcodeIndex == opcode_idx:
        output_models.append(
            self.generate_single_layer_model(self.model, subgraph, op,
                                             opcode_idx))
    return output_models


class PerLayerTestGenerator(generate_test_for_model.TestDataGenerator):

  def generate_tests(self):
    # Collect all target names into a list
    targets = []
    targets_with_path = []
    for model_path in self.model_paths:
      targets.append(model_path.split('/')[-1].split('.')[0])
      targets_with_path.append(
          model_path.split('tflite_micro/')[-1].split('tflite-micro/')
          [-1].split('.')[0])

    template_file_path = os.path.join(TEMPLATE_DIR,
                                      'integration_tests_cc.mako')
    build_template = template.Template(filename=template_file_path)
    with open(self.output_dir + '/integration_tests.cc', 'w') as file_obj:
      key_values_in_template = {
          'targets': targets,
          'targets_with_path': targets_with_path,
          'inputs': self.inputs,
          'input_dtypes': self.input_types,
          'output_dtype': self.output_type
      }
      file_obj.write(build_template.render(**key_values_in_template))

  def generate_build_file(self):
    # Collect all target names into a list
    targets = []
    for model_path in self.model_paths:
      target_name = model_path.split('/')[-1].split('.')[0]
      targets.append(target_name)

    template_file_path = os.path.join(TEMPLATE_DIR, 'BUILD.mako')
    build_template = template.Template(filename=template_file_path)
    with open(self.output_dir + '/BUILD', 'w') as file_obj:
      key_values_in_template = {
          'targets': targets,
          'inputs': self.inputs,
          'input_dtypes': self.input_types,
          'output_dtype': self.output_type
      }
      file_obj.write(build_template.render(**key_values_in_template))


def op_info_from_name(name):
  if 'transpose_conv' in name:
    return [[0, 2], schema_fb.BuiltinOperator.TRANSPOSE_CONV]
  elif 'depthwise_conv' in name:
    return [[0], schema_fb.BuiltinOperator.DEPTHWISE_CONV_2D]
  elif 'conv' in name:
    return [[0], schema_fb.BuiltinOperator.CONV_2D]
  elif 'add' in name:
    return [[0, 1], schema_fb.BuiltinOperator.ADD]
  elif 'sub' in name:
    return [[0, 1], schema_fb.BuiltinOperator.SUB]
  elif 'strided_slice' in name:
    return [[0], schema_fb.BuiltinOperator.STRIDED_SLICE]
  elif 'leaky_relu' in name:
    return [[0], schema_fb.BuiltinOperator.LEAKY_RELU]
  elif 'pad' in name:
    return [[0], schema_fb.BuiltinOperator.PAD]
  else:
    raise RuntimeError(f'Unsupported op: {name}')


FLAGS = flags.FLAGS

flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input TFLite file.')
flags.DEFINE_string('output_dir', None, 'directory to output generated files')

flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_dir')


def main(_):
  model = flatbuffer_utils.read_model(FLAGS.input_tflite_file)
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  inputs, builtin_operator = op_info_from_name(FLAGS.output_dir.split('/')[-1])
  generator = TestModelGenerator(model, FLAGS.output_dir, inputs)
  model_names = generator.generate_models(0, builtin_operator)
  data_generator = PerLayerTestGenerator(FLAGS.output_dir, model_names, inputs)
  data_generator.generate_goldens(builtin_operator)
  data_generator.generate_build_file()
  data_generator.generate_makefile()
  data_generator.generate_tests()
  print(
      f'successfully generated integration tests. Output location: {FLAGS.output_dir}'
  )


if __name__ == '__main__':
  app.run(main)
