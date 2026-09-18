#!/usr/bin/env python3
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
"""Outputs TFLite model structure and size summary using the flatbuffer schema."""

import argparse
import os
import sys

# Add tensorflow/lite/python to sys.path so schema_py_generated can be imported
# directly without requiring a full TensorFlow installation.
_SCHEMA_DIR = os.path.abspath(
  os.path.join(os.path.dirname(__file__), '../../../python')
)
if _SCHEMA_DIR not in sys.path:
  sys.path.insert(0, _SCHEMA_DIR)

import schema_py_generated as schema_fb  # noqa: E402  # pylint: disable=g-import-not-at-top

_TENSOR_TYPE_NAMES = {
  getattr(schema_fb.TensorType, name): name
  for name in dir(schema_fb.TensorType)
  if not name.startswith('_')
}

_BUILTIN_OP_NAMES = {
  getattr(schema_fb.BuiltinOperator, name): name
  for name in dir(schema_fb.BuiltinOperator)
  if not name.startswith('_')
}


def analyze_model(model_path: str) -> None:
  """Reads a .tflite file and prints subgraph, op, tensor, and size summaries."""
  with open(model_path, 'rb') as f:
    buf = bytearray(f.read())

  model = schema_fb.Model.GetRootAsModel(buf, 0)
  print(f'=== {model_path} ===\n')

  num_subgraphs = model.SubgraphsLength()
  print(f"Your TFLite model has '{num_subgraphs}' subgraph(s).")

  op_names = []
  for i in range(model.OperatorCodesLength()):
    opcode = model.OperatorCodes(i)
    code = max(opcode.DeprecatedBuiltinCode(), opcode.BuiltinCode())
    if code == schema_fb.BuiltinOperator.CUSTOM:
      custom_name = (
        opcode.CustomCode().decode('utf-8') if opcode.CustomCode() else 'CUSTOM'
      )
      op_names.append(custom_name)
    else:
      op_names.append(_BUILTIN_OP_NAMES.get(code, f'OP_{code}'))

  for sg_idx in range(num_subgraphs):
    sg = model.Subgraphs(sg_idx)
    ins_str = ', '.join(f'T#{sg.Inputs(i)}' for i in range(sg.InputsLength()))
    outs_str = ', '.join(
      f'T#{sg.Outputs(i)}' for i in range(sg.OutputsLength())
    )
    print(f'\nSubgraph#{sg_idx}({ins_str}) -> [{outs_str}]')
    for op_idx in range(sg.OperatorsLength()):
      op = sg.Operators(op_idx)
      op_name = op_names[op.OpcodeIndex()]
      op_ins = ', '.join(
        f'T#{op.Inputs(i)}'
        for i in range(op.InputsLength())
        if op.Inputs(i) >= 0
      )
      op_outs = ', '.join(
        f'T#{op.Outputs(i)}'
        for i in range(op.OutputsLength())
        if op.Outputs(i) >= 0
      )
      print(f'  Op#{op_idx} {op_name}({op_ins}) -> [{op_outs}]')

    print(f'\nTensors of Subgraph#{sg_idx}')
    for t_idx in range(sg.TensorsLength()):
      t = sg.Tensors(t_idx)
      name = t.Name().decode('utf-8') if t.Name() else ''
      shape = [t.Shape(i) for i in range(t.ShapeLength())]
      ttype = _TENSOR_TYPE_NAMES.get(t.Type(), str(t.Type()))
      buf_idx = t.Buffer()
      buf_obj = (
        model.Buffers(buf_idx) if buf_idx < model.BuffersLength() else None
      )
      buf_len = (
        buf_obj.DataLength() if buf_obj and not buf_obj.DataIsNone() else 0
      )
      ro_str = f' RO {buf_len} bytes, buffer: {buf_idx}' if buf_len > 0 else ''
      print(f'  T#{t_idx}({name}) shape:{shape}, type:{ttype}{ro_str}')

  total_size = len(buf)
  data_size = sum(
    model.Buffers(i).DataLength()
    for i in range(model.BuffersLength())
    if not model.Buffers(i).DataIsNone()
  )
  non_data_size = total_size - data_size
  non_data_pct = 100.0 * non_data_size / total_size if total_size else 0.0
  data_pct = 100.0 * data_size / total_size if total_size else 0.0

  print('\n---------------------------------------------------------------')
  print(f'              Model size: {total_size:10d} bytes')
  print(
    f'    Non-data buffer size: {non_data_size:10d} bytes'
    f' ({non_data_pct:05.2f} %)'
  )
  print(f'  Total data buffer size: {data_size:10d} bytes ({data_pct:05.2f} %)')


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Analyze a .tflite model using flatbuffer schema.'
  )
  parser.add_argument(
    '--model_file',
    type=str,
    required=True,
    help='Path to the .tflite model file.',
  )
  args, _ = parser.parse_known_args()
  analyze_model(args.model_file)


if __name__ == '__main__':
  main()
