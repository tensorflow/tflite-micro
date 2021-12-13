# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""embARC MLI model adaptation tool"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
import shutil

try:
  from tensorflow.lite.python.util import convert_bytes_to_c_source, _convert_model_from_object_to_bytearray, \
    _convert_model_from_bytearray_to_object
except ImportError:
  print('Install TensorFlow package first to use MLI adaptation tool.')
  sys.exit(1)


# Model conversion functions
def convert_c_source_to_bytes(input_cc_file):
  """Converts C++ source file to bytes (immutable).

  Args:
    input_cc_file: A .cc file to process.

  Returns:
    A bytearray corresponding to the input cc file array.
  """
  pattern = re.compile(r'(((0x[0-9a-fA-F]+), ?)+)')
  model_bytearray = bytearray()

  with open(input_cc_file) as file_handle:
    for line in file_handle:
      values_match = pattern.search(line)

      if values_match is None:
        continue

      list_text = values_match.group(1)
      values_text = filter(None, list_text.split(','))

      values = [int(x, base=16) for x in values_text]
      model_bytearray.extend(values)

  return bytes(model_bytearray)


def convert_c_source_to_object(input_cc_file):
  """Converts C++ source file to an object for parsing."""
  with open(input_cc_file, 'r') as model_file:
    include_path, array_name = None, None
    for line in model_file:
      if '#include' in line and not include_path:
        include_path = line.strip('#include ').strip('"\n')
      if re.search(r"\[\].*[=]|\[[1-9][0-9]*\].*[=]", line) and not array_name:
        array_name = re.search(r"\w*(?=\[)", line).group()
      if include_path and array_name:
        break

  model_bytes = convert_c_source_to_bytes(input_cc_file)
  return _convert_model_from_bytearray_to_object(model_bytes), \
         include_path, array_name


def read_model(input_tflite_file):
  """Reads a tflite model as a python object."""

  with open(input_tflite_file, 'rb') as model_file:
    model_bytearray = bytearray(model_file.read())
    return _convert_model_from_bytearray_to_object(model_bytearray)


def write_model(model_object, output_tflite_file, include_path, array_name):
  """Writes the tflite model, a python object, into the output file.

  Args:
    model_object: A tflite model as a python object
    output_tflite_file: Full path name to the output tflite file.
    include_path: Path to model header file
    array_name: name of the array for .cc output

  Raises:
    ValueError: If file is not formatted in .cc or .tflite
  """
  model_bytearray = _convert_model_from_object_to_bytearray(model_object)
  if output_tflite_file.endswith('.cc'):
    mode = 'w'
    converted_model = convert_bytes_to_c_source(data=model_bytearray,
                                                array_name=array_name,
                                                include_path=include_path,
                                                use_tensorflow_license=True)[0]
  elif output_tflite_file.endswith('.tflite'):
    mode = 'wb'
    converted_model = model_bytearray
  else:
    raise ValueError('File format not supported')

  with open(output_tflite_file, mode) as output_file:
    output_file.write(converted_model)


# Helper functions
def transpose_weights(tensor, buffer, transpose_shape):
  """Transposes weights to embARC MLI format according to transpose_shape

  Args:
    tensor: A tensor to process
    buffer: A buffer relevant to the tensor
    transpose_shape: Target shape.
  """
  buffer.data = buffer.data \
    .reshape(tensor.shape) \
    .transpose(transpose_shape) \
    .flatten()

  tensor.shape = tensor.shape[transpose_shape]

  tensor.quantization.quantizedDimension = \
    transpose_shape.index(tensor.quantization.quantizedDimension)


# Layer-specific adaptation functions
def adapt_conv(operator, tensors, buffers):
  """Adapts weights tensors of convolution layers

  Args:
    operator: Operator index
    tensors: Model tensors dict
    buffers: Model buffers dict
  """
  transpose_weights(tensors[operator.inputs[1]],
                    buffers[tensors[operator.inputs[1]].buffer], [1, 2, 3, 0])


def adapt_dw(operator, tensors, _buffers):
  """Adapts weights tensors of depthwise convolution layers

  Args:
    operator: Operator index
    tensors: Model tensors dict
    _buffers: Model buffers dict
  """
  tensors[operator.inputs[1]].shape = \
    tensors[operator.inputs[1]].shape[[1, 2, 0, 3]]


def adapt_fc(operator, tensors, buffers):
  """Adapts weights tensors of fully connected layers

  Args:
    operator: Operator index
    tensors: Model tensors dict
    buffers: Model buffers dict
  """
  transpose_weights(tensors[operator.inputs[1]],
                    buffers[tensors[operator.inputs[1]].buffer], [1, 0])


# Op_codes that require additional adaptation for MLI
adapt_op_codes = {
    3: adapt_conv,  # CONV_2D
    4: adapt_dw,  # DEPTHWISE_CONV_2D
    9: adapt_fc  # FULLY_CONNECTED
}


def adapt_model_to_mli(model):
  """Adapts weights of the model to embARC MLI layout

  Args:
    model: TFLite model object
  """
  op_codes = [
      op_code.builtinCode
      if op_code.builtinCode != 0 else op_code.deprecatedBuiltinCode
      for op_code in model.operatorCodes
  ]
  for subgraph in model.subgraphs:
    for operator in subgraph.operators:
      try:
        adapt_op_codes[op_codes[operator.opcodeIndex]] \
          (operator, subgraph.tensors, model.buffers)
      except KeyError:
        continue


def main(argv):
  try:
    if len(sys.argv) == 3:
      tflite_input = argv[1]
      tflite_output = argv[2]
    elif len(sys.argv) == 2:
      tflite_input = argv[1]
      tflite_output = argv[1]
  except IndexError:
    print("Usage: %s <input cc/tflite> <output cc/tflite>" % (argv[0]))
  else:
    if tflite_input == tflite_output:
      path, filename = os.path.split(tflite_input)
      try:
        shutil.copyfile(tflite_input, path + '/orig_' + filename)
      except OSError as err:
        print('Error while creating backup file:', err)
    if tflite_input.endswith('.cc'):
      model, include_path, array_name = convert_c_source_to_object(
          tflite_input)
    elif tflite_input.endswith('.tflite'):
      model = read_model(tflite_input)
      include_path = ''
      array_name = os.path.split(tflite_output)[1].split('.')[0]
    else:
      raise ValueError('File format not supported')

    adapt_model_to_mli(model)
    write_model(model, tflite_output, include_path, array_name)

    print('Model was adapted to be used with embARC MLI.')


if __name__ == "__main__":
  main(sys.argv)