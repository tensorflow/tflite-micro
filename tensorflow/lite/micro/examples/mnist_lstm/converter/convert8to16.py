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
"""An experimental tool to convert a int8 activation, int8 weight LSTM based model to int16 activation, int8 weight

Steps: 
1. Convert the trained model to int8 using the TFLite converter. See https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization
2. Use this tool to convert the int8 model to int16.
3. Check if the converted model match the expectation (e.g., read the conversion printout, perform inference tests)

The conversion process: 
1. Convert the ops specified in _OP_CONVERSION_REGISTRATION using the registered function. Bias type conversion (int32 to int64) only happens here. 
2. Convert all non-constant tensors with int8 type to int16 (and fix the quantization parameters)

Run:
`bazel build tensorflow/lite/micro/examples/mnist_lstm:convert8to16`
`bazel-bin/tensorflow/lite/micro/examples/mnist_lstm/convert8to16
--int8_model_path=".tflite file path"` --save_path="save path"

CAVEAT: 
1. Use this tool ONLY for models that contain the LSTM layer. All other models should use the standard tflite conversion process.
2. This is an experimental tool. ALWAYS check if the converted model matches your expectation
3. Add the custom op conversion function for complex ops (e.g., convolution). 

"""
import os

import numpy as np
from absl import app
from absl import flags
from absl import logging

from tflite_micro.tensorflow.lite.tools import flatbuffer_utils
import converter_utils as utils

FLAGS = flags.FLAGS

flags.DEFINE_string("int8_model_path", "../trained_lstm_quant.tflite",
                    "the int8 model path.")
flags.DEFINE_string("save_path", "/tmp/8to16model.tflite",
                    "path to save the converted model.")

# key: BuiltinOperator; Val: the conversion function (see tensorflow/lite/schema/schema.fbs)
_OP_CONVERSION_REGISTRATION = {9: utils.convert_fully_connected,
                               44: utils.convert_unidirectional_sequence_lstm,
                               25: utils.convert_softmax}



class Int8ToInt16Converter:
  """Convert an int8 activation model to int16"""
  def __init__(self, int8_model):
    """Initialize the int8 to int16 converter.

    Args:
      int8_model: flatbuffer python object
    """
    self.model = int8_model
    self.remaining_tensors = set()
    for subgraph in self.model.subgraphs:
      for tensor in subgraph.tensors:
        self.remaining_tensors.add(tensor)

  @classmethod
  def from_file(self, model_path):
    """Instantiates a converter from a int8 quantized .tflite filepath.

    Args:
      model_path: Filepath to the .tflite model

    Returns:
      An Int8ToInt16Converter instance
    """
    int8_model = flatbuffer_utils.read_model(model_path)
    return Int8ToInt16Converter(int8_model)
    
  @classmethod
  def from_bytes(self, bytearray):
    """Instantiates a converter from a int8 quantized .tflite bytearray.

    Args:
      bytearray: Content of the .tflite model

    Returns:
      An Int8ToInt16Converter instance
    """
    int8_model = flatbuffer_utils.convert_bytearray_to_object(bytearray)
    return Int8ToInt16Converter(int8_model)

  def _remove_tensor(self, tensor):
    """Remove tensor from the tensor pool"""
    if tensor in self.remaining_tensors:
      self.remaining_tensors.remove(tensor)

  def _remove_op_tensors(self, tensors, op):
    """Remove tensors in an operator from the tensor pool

    Args:
        tensors: tensors in the subgraph
        op : the operator
    """
    for id in op.inputs:
      self._remove_tensor(tensors[id])
    for id in op.outputs:
      self._remove_tensor(tensors[id])

  def _convert_ops(self):
    """Convert all ops registered in _OP_CONVERSION_REGISTRATION from int8 to int16 (activation type)"""
    op_codes = self.model.operatorCodes
    for subgraph in self.model.subgraphs:
      tensors = subgraph.tensors
      for op in subgraph.operators:
        op_code = op_codes[op.opcodeIndex].builtinCode
        if op_code in _OP_CONVERSION_REGISTRATION:
          logging.info(f"Convert operator {flatbuffer_utils.opcode_to_name(self.model,op.opcodeIndex)}")
          _OP_CONVERSION_REGISTRATION[op_code](tensors, self.model.buffers, op)
          self._remove_op_tensors(tensors, op)

  def _change_tensor_activation_type(self):
    """Change all remaining tensor types from int8 to int16"""
    for subgraph in self.model.subgraphs:
      for tensor in subgraph.tensors:
        if (
            (tensor in self.remaining_tensors)
            and (utils.TENSOR_CODE_TYPE[tensor.type] == np.int8)
            and ("const" not in str(tensor.name))
        ):
          utils.change_activation_tensor_8to16(tensor)
          self._remove_tensor(tensor)

  def convert(self):
    '''
    The conversion process has two phase:
    1. Go through the registered ops and perform the custom op transformation 
    2. Go through the rest of tensors and convert int8 non-const tensor to int16
    '''

    logging.info("Reset OPS")
    self._convert_ops()
    logging.info(
        "Set Remaining Activation Types"
    )
    self._change_tensor_activation_type()
    logging.info("Remaining Tensors:")
    for tensor in self.remaining_tensors:
       logging.info(f"{tensor.name}, tensor type {flatbuffer_utils.type_to_name(tensor.type)}")

  def save_model(self, output_path):
    """Save the requantized model to a specificed location."""
    flatbuffer_utils.write_model(self.model, output_path)

def main(_):
  if not os.path.exists(FLAGS.int8_model_path):
    raise ValueError(
        "Model file does not exist. Please check the .tflite model path.")
  converter = Int8ToInt16Converter.from_file(FLAGS.int8_model_path)
  converter.convert()
  converter.save_model(FLAGS.save_path)

if __name__ == "__main__":
  app.run(main)