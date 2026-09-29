# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Randomize all weights in a tflite file."""

import argparse

from tflite_micro.tensorflow.lite.tools import flatbuffer_utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input_tflite_file',
    required=True,
    help='Full path name to the input TFLite file.',
  )
  parser.add_argument(
    '--output_tflite_file',
    required=True,
    help='Full path name to the output randomized TFLite file.',
  )
  parser.add_argument(
    '--buffers_to_skip',
    action='append',
    type=int,
    default=[],
    help=(
      'Buffer indices in the TFLite model to be skipped, i.e., to be left'
      ' unmodified.'
    ),
  )
  parser.add_argument(
    '--ops_to_skip',
    action='append',
    default=[],
    help='Ops in the TFLite model to be skipped / unmodified.',
  )
  parser.add_argument(
    '--ops_operands_to_skip',
    action='append',
    default=[],
    help=(
      'Op operand indices in the TFLite model to be skipped / unmodified. It'
      ' should be specified in the format'
      ' <op_name>:<operand_index>[,<operand_index>]. For example,'
      ' TRANSPOSE_CONV:0,2 stands for skipping the TRANSPOSE_CONV operands'
      ' indexed 0 and 2'
    ),
  )
  parser.add_argument(
    '--random_seed',
    type=int,
    default=0,
    help='Input to the random number generator.',
  )
  args, _ = parser.parse_known_args()

  buffers_to_skip = list(args.buffers_to_skip)
  ops_to_skip = [op.upper() for op in args.ops_to_skip]
  ops_operands_to_skip = {}
  for op_operands_to_skip in args.ops_operands_to_skip:
    op_name, indices = op_operands_to_skip.split(':')
    op_name_upper = op_name.upper()
    if op_name_upper in ops_operands_to_skip:
      raise ValueError(
        'Indices for the same op must be specified only once multiple'
        f' specification for op {op_name}.'
      )
    ops_operands_to_skip[op_name_upper] = list(map(int, indices.split(',')))

  model = flatbuffer_utils.read_model(args.input_tflite_file)

  # Add in buffers for ops in ops_to_skip or ops_operands_to_skip to the list of
  # skipped buffers.
  for graph in model.subgraphs:
    for op in graph.operators:
      op_name = flatbuffer_utils.opcode_to_name(model, op.opcodeIndex)
      op_name_upper = op_name.upper()
      if op_name_upper in ops_to_skip:
        for input_idx in op.inputs:
          buffers_to_skip.append(graph.tensors[input_idx].buffer)
      if op_name_upper in ops_operands_to_skip:
        for operand_idx in ops_operands_to_skip[op_name_upper]:
          buffers_to_skip.append(graph.tensors[op.inputs[operand_idx]].buffer)

  flatbuffer_utils.randomize_weights(model, args.random_seed, buffers_to_skip)
  flatbuffer_utils.write_model(model, args.output_tflite_file)


if __name__ == '__main__':
  main()
