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
r"""Strips all nonessential strings from a TFLite file."""

from absl import app
from absl import flags
import struct
import copy

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input TFLite file.')
flags.DEFINE_string('output_tflite_file', None,
                    'Full path name to the output stripped TFLite file.')

flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_tflite_file')

def convert_buffers_to_float32(model):
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.type == schema_fb.TensorType.FLOAT16:
        tensor.type = schema_fb.TensorType.FLOAT32
        buf = model.buffers[tensor.buffer]
        new_buf = []
        for i in range(0, len(buf.data), 2):
          byte1 = buf.data[i]
          byte2 = buf.data[i+1]
          fraction = byte1 / 1024.0 + (byte2 & 0x03) / 4 + 1.0
          exponent = ((byte2 & 0x7c) >> 2) - 15
          sign = (byte2 >> 7)
          if sign == 0:
            sign = 1
          else:
            sign = -1
          exp_part = 0
          if (exponent >= 0):
            exp_part = 2 ** exponent
          else:
            exp_part = 1 / (2 ** abs(exponent))
          value = fraction * exp_part * sign
          byte_array = struct.pack('f', value)
          new_buf.append(int(byte_array[0]))
          new_buf.append(int(byte_array[1]))
          new_buf.append(int(byte_array[2]))
          new_buf.append(int(byte_array[3]))

        buf.data = new_buf

  return model
          

def replace_all_tensor_instances(subgraph, old_idx, new_idx):
  subgraph.inputs = [new_idx if idx == old_idx else idx for idx in subgraph.inputs]
  subgraph.outputs = [new_idx if idx == old_idx else idx for idx in subgraph.outputs]

  for op in subgraph.operators:
    op.inputs = [new_idx if idx == old_idx else idx for idx in op.inputs]
    op.outputs = [new_idx if idx == old_idx else idx for idx in op.outputs]


def remove_quant_dequant(model):
  dequantize_opcode = -1
  for idx, opcode in enumerate(model.operatorCodes):
    print(opcode.deprecatedBuiltinCode)
    if opcode.deprecatedBuiltinCode == schema_fb.BuiltinOperator.DEQUANTIZE:
      dequantize_opcode = idx
  for subgraph in model.subgraphs:
    new_op_list = []
    for op in subgraph.operators:
      if op.opcodeIndex == dequantize_opcode:
        # There is only one input / output to dequantize.
        input_idx = op.inputs[0]
        output_idx = op.outputs[0]
        replace_all_tensor_instances(subgraph, output_idx, input_idx)
      else:
        new_op_list.append(copy.deepcopy(op))
    subgraph.operators = new_op_list
  return model


def main(_):
  model = flatbuffer_utils.read_model_with_mutable_tensors(FLAGS.input_tflite_file)
  model = convert_buffers_to_float32(model)
  model = remove_quant_dequant(model)
  flatbuffer_utils.write_model(model, FLAGS.output_tflite_file)


if __name__ == '__main__':
  app.run(main)
