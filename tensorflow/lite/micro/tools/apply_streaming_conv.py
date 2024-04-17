# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Updates a TFLite model with the custom Streaming Conv2d operator."""

from absl import app
from absl import flags
from absl import logging

from tflite_micro.tensorflow.lite.micro.tools import model_transforms_utils
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils

_INPUT_MODEL = flags.DEFINE_string(
    "input_model",
    None,
    ".tflite input model path",
    required=True,
)

_OUTPUT_MODEL = flags.DEFINE_string(
    "output_model",
    None,
    ".tflite output model path",
    required=True,
)

_CONV_2D = schema_fb.BuiltinOperator.CONV_2D
_STREAMING_CONV_2D = schema_fb.BuiltinOperator.TEMP_STREAMING_CONV_2D
_CONCAT = schema_fb.BuiltinOperator.CONCATENATION
_STRIDED_SLICE = schema_fb.BuiltinOperator.STRIDED_SLICE
_PAD = schema_fb.BuiltinOperator.PAD
_VAR_HANDLE = schema_fb.BuiltinOperator.VAR_HANDLE
_READ_VAR = schema_fb.BuiltinOperator.READ_VARIABLE
_ASSIGN_VARIABLE = schema_fb.BuiltinOperator.ASSIGN_VARIABLE


class StreamingConvConverter:
  """ Converts an existing kws-streaming CNN model to use custom StreamingConv2D
      operators. """

  def __init__(self, model):
    self.model = model
    self.remaining_tensors = set()
    for subgraph in self.model.subgraphs:
      for tensor in subgraph.tensors:
        self.remaining_tensors.add(tensor)

  @classmethod
  def from_file(self, model_path):
    model = flatbuffer_utils.read_model(model_path)
    return StreamingConvConverter(model)

  def convert(self):
    logging.info("Converting...")
    logging.info("Adding StreamConv opcode.")
    self._add_streaming_conv2d_opcode()
    logging.info("Replacing Conv2d operators with StreamingConv2d.")
    self._replace_conv2d_with_streaming_conv2d()
    logging.info("Done.")

  def save_model(self, output_path):
    flatbuffer_utils.write_model(self.model, output_path)
    model_transforms_utils.tflite_flatbuffer_align(output_path, output_path)

  def _add_streaming_conv2d_opcode(self):
    op_code = schema_fb.OperatorCodeT()
    op_code.deprecatedBuiltinCode = schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
    op_code.builtinCode = _STREAMING_CONV_2D
    self.model.operatorCodes.append(op_code)
    self.streaming_conv_opcode_index = len(self.model.operatorCodes) - 1

  def _replace_conv2d_with_streaming_conv2d(self):
    for subgraph in self.model.subgraphs:
      operators_to_remove = []
      # Tensor -> Operator that produces it.
      producer = {}
      # Tensor -> Operators that consume it.
      consumers = {}
      for operator in subgraph.operators:
        if operator.outputs is not None:
          for output_tensor in operator.outputs:
            producer[output_tensor] = operator
        if operator.inputs is not None:
          for input_tensor in operator.inputs:
            if input_tensor not in consumers:
              consumers[input_tensor] = []
            consumers[input_tensor].append(operator)

      for operator in subgraph.operators:
        if (self._is_opcode(operator, _CONV_2D)
            and self._is_opcode(producer[operator.inputs[0]], _CONCAT)):
          conv2d_op = operator
          concat_op = producer[conv2d_op.inputs[0]]
          read_var_op = producer[concat_op.inputs[0]]
          pad_op = producer[concat_op.inputs[1]]
          var_handle_op = producer[read_var_op.inputs[0]]

          assert (self._is_opcode(concat_op, _CONCAT))
          assert (self._is_opcode(read_var_op, _READ_VAR))
          assert (self._is_opcode(pad_op, _PAD))
          assert (self._is_opcode(var_handle_op, _VAR_HANDLE))

          conv2d_op.inputs[0] = pad_op.outputs[0]
          conv2d_op.opcodeIndex = self.streaming_conv_opcode_index
          operators_to_remove.extend((concat_op, var_handle_op))
          operators_to_remove.extend(consumers[var_handle_op.outputs[0]])
          concat_consumers = consumers[concat_op.outputs[0]]
          concat_consumers.remove(conv2d_op)
          operators_to_remove.extend(concat_consumers)

      for operator in operators_to_remove:
        subgraph.operators.remove(operator)

  def _is_opcode(self, operator, code):
    opcode_index = operator.opcodeIndex
    op_code = self.model.operatorCodes[opcode_index]
    if op_code.deprecatedBuiltinCode == schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
      return op_code.builtinCode == code
    else:
      return op_code.deprecatedBuiltinCode == code


def main(_) -> None:
  input_model_path = _INPUT_MODEL.value
  output_model_path = _OUTPUT_MODEL.value
  logging.info(
      "Replacing streaming conv2d pattern with StreamingConv2d op:"
      "\n  input_model: %s\n  output_model: %s", input_model_path,
      output_model_path)
  converter = StreamingConvConverter.from_file(input_model_path)
  converter.convert()
  converter.save_model(output_model_path)


if __name__ == "__main__":
  app.run(main)
