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
"""
Usage:
  bazel run tensorflow/lite/micro/tools:relocate_read_variable -- \\
      --input=<input.tflite> --output=<output.tflite>
"""

import model_facade

from tensorflow.lite.python import schema_py_generated as tflite

from absl import app
from absl import flags
from pathlib import Path
from typing import List, Set, Dict, Tuple

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='input',
    default='',
    help='path for the .tflite input file',
)

flags.DEFINE_string(
    name='output',
    default='',
    help='path for the .tflite output file',
)

VarHandleId = int
VarHandles = Set[VarHandleId]
TensorIndex = int
SubgraphIndex = int

VarHandleByName = Dict[Tuple[str | None, str], VarHandleId]
"""VarHandleByName
{ ( container_name | None, resource_name ) : var_handle_id }
"""

VarHandleByTensor = Dict[Tuple[SubgraphIndex, TensorIndex], VarHandleId]
"""VarHandleByTensor
{ ( subgraph_index, tensor_index ) : var_handle_id }
"""

ReadVarOps = List[Dict[int, model_facade.Operator]]

ConcatOps = List[Dict[int, model_facade.Operator]]


class Context:
  """
  Context:
  """

  def __init__(self, model: model_facade.Model) -> None:
    self._model = model
    self._current_op_stack: List[model_facade.Operator] = []
    self._reordered_operators: List[List[model_facade.Operator]] = [[]] * len(
        model.subgraphs)
    self._subgraph_processed: List[bool] = [False] * len(model.subgraphs)
    self._subgraph_modified_vars: List[VarHandles] = [set()] * len(
        model.subgraphs)
    self._subgraph_read_var_ops: ReadVarOps = [{}] * len(model.subgraphs)
    self._subgraph_concat_ops: ConcatOps = [{}] * len(model.subgraphs)
    self._var_handles_by_name: VarHandleByName = {}
    self._var_handles_by_tensor: VarHandleByTensor = {}
    self._current_var_handle_id: VarHandleId = 0

  @property
  def model(self):
    return self._model

  def current_op(self) -> model_facade.Operator:
    return self._current_op_stack[-1]

  def push_current_op(self, op: model_facade.Operator) -> None:
    self._current_op_stack.append(op)

  def pop_current_op(self) -> None:
    _ = self._current_op_stack.pop()

  def append_to_reordered_operations(self, op: model_facade.Operator) -> None:
    subgraph_index: int = op.subgraph.index
    new_op = model_facade.Operator(
        op.operator, len(self._reordered_operators[subgraph_index]),
        op.subgraph)
    self._reordered_operators[subgraph_index].append(new_op)

  def reordered_operations(
      self, subgraph_index: SubgraphIndex) -> List[model_facade.Operator]:
    return self._reordered_operators[subgraph_index]

  def is_subgraph_processed(self, subgraph_index: SubgraphIndex) -> bool:
    return self._subgraph_processed[subgraph_index]

  def mark_subgraph_processed(self, subgraph_index: SubgraphIndex) -> None:
    self._subgraph_processed[subgraph_index] = True

  def subgraph_var_handles(self, subgraph_index: SubgraphIndex) -> VarHandles:
    return self._subgraph_modified_vars[subgraph_index]

  def set_subgraph_var_handles(self, subgraph_index: SubgraphIndex,
                               handles: VarHandles) -> None:
    self._subgraph_modified_vars[subgraph_index] = handles

  def add_read_var_op(self, op: model_facade.Operator) -> None:
    assert op.builtin_opcode == tflite.BuiltinOperator.READ_VARIABLE
    key: TensorIndex = op.outputs_indices[0]
    self._subgraph_read_var_ops[op.subgraph.index][key] = op

  def remove_read_var_op(self, op: model_facade.Operator) -> None:
    assert op.builtin_opcode == tflite.BuiltinOperator.READ_VARIABLE
    key: TensorIndex = op.outputs_indices[0]
    del self._subgraph_read_var_ops[op.subgraph.index][key]

  def get_read_var_op_by_tensor(
      self, tensor_index: TensorIndex,
      subgraph_index: SubgraphIndex) -> model_facade.Operator | None:
    return self._subgraph_read_var_ops[subgraph_index].get(tensor_index, None)

  def get_read_var_op_by_handle(
      self, resource_tensor_index: TensorIndex,
      subgraph_index: SubgraphIndex) -> List[model_facade.Operator]:
    result: List[model_facade.Operator] = []
    var_handle_id = self.get_var_handle(subgraph_index, resource_tensor_index)
    for op in self._subgraph_read_var_ops[subgraph_index].values():
      if self.get_var_handle(op.subgraph.index,
                             op.inputs_indices[0]) == var_handle_id:
        result.append(op)
    return result

  def add_concat_op(self, op: model_facade.Operator) -> None:
    assert op.builtin_opcode == tflite.BuiltinOperator.CONCATENATION
    key: TensorIndex = op.outputs_indices[0]
    self._subgraph_concat_ops[op.subgraph.index][key] = op

  def remove_concat_op(self, op: model_facade.Operator) -> None:
    assert op.builtin_opcode == tflite.BuiltinOperator.CONCATENATION
    key: TensorIndex = op.outputs_indices[0]
    del self._subgraph_concat_ops[op.subgraph.index][key]

  def get_concat_op_by_tensor(
      self, tensor_index: TensorIndex,
      subgraph_index: SubgraphIndex) -> model_facade.Operator | None:
    return self._subgraph_concat_ops[subgraph_index].get(tensor_index, None)

  def create_var_handle(self, container_name: str | None, resource_name: str,
                        subgraph_index: SubgraphIndex,
                        resource_tensor_index: TensorIndex) -> VarHandleId:
    key = (container_name, resource_name)
    var_handle_id = self._var_handles_by_name.get(key)
    if var_handle_id is None:
      var_handle_id = self._current_var_handle_id
      self._current_var_handle_id += 1
      self._var_handles_by_name[key] = var_handle_id

    self.add_var_handle(subgraph_index, resource_tensor_index, var_handle_id)

    return var_handle_id

  def get_var_handle(self, subgraph_index: SubgraphIndex,
                     resource_tensor_index: TensorIndex) -> VarHandleId:
    return self._var_handles_by_tensor[(subgraph_index, resource_tensor_index)]

  def add_var_handle(self, subgraph_index: SubgraphIndex,
                     resource_tensor_index: TensorIndex,
                     var_handle_id: VarHandleId) -> None:
    key = (subgraph_index, resource_tensor_index)
    assert self._var_handles_by_tensor.get(key, None) is None
    self._var_handles_by_tensor[key] = var_handle_id


# Begin global methods


def process_operator_var_handle(context: Context) -> VarHandles:
  op = context.current_op()
  assert op.builtin_options_type == tflite.BuiltinOptions.VarHandleOptions
  assert op.builtin_options is not None
  container_name: str = op.builtin_options.container
  resource_name: str = op.builtin_options.sharedName
  var_handle_id = context.create_var_handle(container_name, resource_name,
                                            op.subgraph.index,
                                            op.outputs_indices[0])
  context.append_to_reordered_operations(op)
  return set([var_handle_id])


def process_operator_assign_variable(context: Context) -> VarHandles:
  assign_op = context.current_op()
  pending_concat_op = context.get_concat_op_by_tensor(
      assign_op.inputs_indices[1], assign_op.subgraph.index)
  assert pending_concat_op is None
  read_var_op = context.get_read_var_op_by_tensor(assign_op.inputs_indices[1],
                                                  assign_op.subgraph.index)
  if read_var_op is not None:
    context.append_to_reordered_operations(read_var_op)
    context.remove_read_var_op(read_var_op)

  for read_var_op in context.get_read_var_op_by_handle(
      assign_op.inputs_indices[0], assign_op.subgraph.index):
    context.append_to_reordered_operations(read_var_op)
    context.remove_read_var_op(read_var_op)

  context.append_to_reordered_operations(assign_op)
  return set()


def process_operator_read_variable(context: Context) -> VarHandles:
  context.add_read_var_op(context.current_op())
  return set()


def process_operator_call_once(context: Context) -> VarHandles:
  assert False
  return set()


def process_operator_if(context: Context) -> VarHandles:
  assert False
  return set()


def process_operator_while(context: Context) -> VarHandles:
  assert False
  return set()


def process_operator_concatenation(context: Context) -> VarHandles:
  context.add_concat_op(context.current_op())
  return set()


def process_operator(context: Context) -> VarHandles:
  op = context.current_op()
  if op.builtin_opcode == tflite.BuiltinOperator.VAR_HANDLE:
    return process_operator_var_handle(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.ASSIGN_VARIABLE:
    return process_operator_assign_variable(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.READ_VARIABLE:
    return process_operator_read_variable(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.CONCATENATION:
    return process_operator_concatenation(context)
  else:
    for tensor_input in op.inputs_indices:
      concat_op = context.get_concat_op_by_tensor(tensor_input,
                                                  op.subgraph.index)
      if concat_op is not None:
        for concat_tensor_input in concat_op.inputs_indices:
          pending_concat_op = context.get_concat_op_by_tensor(
              concat_tensor_input, op.subgraph.index)
          assert pending_concat_op is None
          read_var_op = context.get_read_var_op_by_tensor(
              concat_tensor_input, op.subgraph.index)
          if read_var_op is not None:
            context.append_to_reordered_operations(read_var_op)
            context.remove_read_var_op(read_var_op)

        context.append_to_reordered_operations(concat_op)
        context.remove_concat_op(concat_op)

      read_var_op = context.get_read_var_op_by_tensor(tensor_input,
                                                      op.subgraph.index)
      if read_var_op is not None:
        context.append_to_reordered_operations(read_var_op)
        context.remove_read_var_op(read_var_op)
    context.append_to_reordered_operations(op)

  return set()


def process_subgraph(context: Context, subgraph_index: int) -> VarHandles:
  if context.is_subgraph_processed(subgraph_index):
    return context.subgraph_var_handles(subgraph_index)

  var_handles: VarHandles = set()
  subgraph: model_facade.Subgraph = context.model.subgraphs[subgraph_index]
  op: model_facade.Operator

  for op in subgraph.operators:
    context.push_current_op(op)
    var_handles_processed: VarHandles = process_operator(context)
    var_handles.update(var_handles_processed)
    context.pop_current_op()

  operators: List[tflite.OperatorT] = []
  for op in context.reordered_operations(subgraph_index):
    operators.append(op.operator)
  context.model.root.subgraphs[subgraph_index].operators = operators

  context.mark_subgraph_processed(subgraph_index)
  context.set_subgraph_var_handles(subgraph_index, var_handles)

  return var_handles


def main(_):
  input_path = Path(FLAGS.input)
  output_path = Path(FLAGS.output)

  with open(input_path, 'rb') as file:
    buffer = bytes(file.read())
  input_model: model_facade.Model = model_facade.read(buffer)

  context = Context(input_model)
  _ = process_subgraph(context, 0)

  output_model: bytearray = input_model.pack()
  with open(output_path, 'wb') as file:
    file.write(output_model)


if __name__ == '__main__':
  app.run(main)
