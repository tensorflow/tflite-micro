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
r"""
*** EXPERIMENTAL ***
This is an experimental tool and is subject to change at any time

This tool will allow reordering of the command line specified operators.
The reordered operators will be moved within their subgraph, such that they
are more closely colocated to another operator which consumes their output
tensor.

The output model will be properly aligned as per the .tflite flatbuffer schema.

Usage:
  bazel run tensorflow/lite/micro/tools:reorder_ops -- \
      --input=<input.tflite> \
      --output=<output.tflite> \
      --ops=<operator_name[,operator_name]...>
"""

from tflite_micro.tensorflow.lite.micro.compression import model_facade
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite
from tflite_micro.tensorflow.lite.micro.tools import model_transforms_utils

from absl import app
from absl import flags
from pathlib import Path
from typing import List, Set, Dict, Tuple

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='input',
    default=None,
    help='path for the .tflite input file',
    required=True,
)

flags.DEFINE_string(
    name='output',
    default=None,
    help='path for the .tflite output file',
    required=True,
)

flags.DEFINE_list(
    name='ops',
    default=[],
    help='comma separated names of operators to reorder (case insensitive)',
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

PendingOps = List[Dict[TensorIndex, model_facade._Operator]]
"""PendingOps
[ { output_tensor_index : operator }]
"""


class Context:
  """
  Context:

  The Context holds the stack of operators currently being processed,
  the list of pending operations which may be relocated,
  and the list of reordered operations representing the new subgraph(s)
  """

  def __init__(self, model: model_facade._Model, ops: List[int]) -> None:
    self._model = model
    self._current_op_stack: List[model_facade._Operator] = []
    self._reordered_operators: List[List[model_facade._Operator]] = [[]] * len(
        model.subgraphs)
    self._subgraph_processed: List[bool] = [False] * len(model.subgraphs)
    self._subgraph_modified_vars: List[VarHandles] = [set()] * len(
        model.subgraphs)
    self._pending_ops: PendingOps = [{}] * len(model.subgraphs)
    self._var_handles_by_name: VarHandleByName = {}
    self._var_handles_by_tensor: VarHandleByTensor = {}
    self._current_var_handle_id: VarHandleId = 0
    self._ops_to_reorder: List[int] = ops

  @property
  def model(self):
    return self._model

  def current_op(self) -> model_facade._Operator:
    return self._current_op_stack[-1]

  def push_current_op(self, op: model_facade._Operator) -> None:
    self._current_op_stack.append(op)

  def pop_current_op(self) -> None:
    _ = self._current_op_stack.pop()

  def append_to_reordered_operations(self, op: model_facade._Operator) -> None:
    subgraph_index: int = op.subgraph.index
    new_op = model_facade._Operator(
        op.operator, len(self._reordered_operators[subgraph_index]),
        op.subgraph)
    self._reordered_operators[subgraph_index].append(new_op)

  def reordered_operations(
      self, subgraph_index: SubgraphIndex) -> List[model_facade._Operator]:
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

  def add_pending_op(self, op: model_facade._Operator) -> None:
    assert len(op.outputs_indices) == 1
    key: TensorIndex = op.outputs_indices[0]
    assert self._pending_ops[op.subgraph.index].get(key) is None
    self._pending_ops[op.subgraph.index][key] = op

  def remove_pending_op(self, op: model_facade._Operator) -> None:
    key: TensorIndex = op.outputs_indices[0]
    assert self._pending_ops[op.subgraph.index][key].index == op.index
    del self._pending_ops[op.subgraph.index][key]

  def get_pending_op(
      self, tensor_index: TensorIndex,
      subgraph_index: SubgraphIndex) -> model_facade._Operator | None:
    return self._pending_ops[subgraph_index].get(tensor_index)

  def get_pending_read_var_ops_by_handle(
      self, var_handle_id: VarHandleId,
      subgraph_index: SubgraphIndex) -> List[model_facade._Operator]:
    result: List[model_facade._Operator] = []
    for op in self._pending_ops[subgraph_index].values():
      if op.builtin_opcode != tflite.BuiltinOperator.READ_VARIABLE:
        continue
      if self.get_var_handle(op.subgraph.index,
                             op.inputs_indices[0]) == var_handle_id:
        result.append(op)
    return result

  def can_be_pending_op(self, op: model_facade._Operator) -> bool:
    return (op.builtin_opcode in self._ops_to_reorder
            and op.outputs_indices is not None
            and len(op.outputs_indices) == 1)

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
    assert self._var_handles_by_tensor.get(key) is None
    self._var_handles_by_tensor[key] = var_handle_id


# Begin global methods


def process_operator_var_handle(context: Context) -> None:
  op = context.current_op()
  assert op.builtin_options_type == tflite.BuiltinOptions.VarHandleOptions
  assert op.builtin_options is not None
  container_name: str = op.builtin_options.container
  resource_name: str = op.builtin_options.sharedName
  _ = context.create_var_handle(container_name, resource_name,
                                op.subgraph.index, op.outputs_indices[0])
  if context.can_be_pending_op(op):
    context.add_pending_op(op)
  else:
    context.append_to_reordered_operations(op)


def process_operator_assign_variable(context: Context) -> VarHandles:
  assign_op = context.current_op()
  var_handle_id = context.get_var_handle(assign_op.subgraph.index,
                                         assign_op.inputs_indices[0])
  for read_var_op in context.get_pending_read_var_ops_by_handle(
      var_handle_id, assign_op.subgraph.index):
    context.remove_pending_op(read_var_op)
    process_read_var_pending_ops(context, read_var_op)

  process_pending_ops(context)

  return set([var_handle_id])


def process_operator_call_once(context: Context) -> VarHandles:
  op = context.current_op()
  assert op.builtin_options_type == tflite.BuiltinOptions.CallOnceOptions
  assert op.builtin_options is not None
  subgraph_index: int = op.builtin_options.subgraph
  var_handles: VarHandles = process_subgraph(context, subgraph_index)
  for var_handle_id in var_handles:
    for read_var_op in context.get_pending_read_var_ops_by_handle(
        var_handle_id, op.subgraph.index):
      context.remove_pending_op(read_var_op)
      process_read_var_pending_ops(context, read_var_op)

  context.append_to_reordered_operations(op)

  return var_handles


def process_operator_if(context: Context) -> VarHandles:
  op = context.current_op()
  assert op.builtin_options_type == tflite.BuiltinOptions.IfOptions
  assert op.builtin_options is not None
  then_subgraph_index: int = op.builtin_options.thenSubgraphIndex
  else_subgraph_index: int = op.builtin_options.elseSubgraphIndex
  var_handles: VarHandles = process_subgraph(context, then_subgraph_index)
  var_handles |= process_subgraph(context, else_subgraph_index)
  for var_handle_id in var_handles:
    for read_var_op in context.get_pending_read_var_ops_by_handle(
        var_handle_id, op.subgraph.index):
      context.remove_pending_op(read_var_op)
      process_read_var_pending_ops(context, read_var_op)

  process_pending_ops(context)

  return var_handles


def process_operator_while(context: Context) -> VarHandles:
  op = context.current_op()
  assert op.builtin_options_type == tflite.BuiltinOptions.WhileOptions
  assert op.builtin_options is not None
  cond_subgraph_index: int = op.builtin_options.condSubgraphIndex
  body_subgraph_index: int = op.builtin_options.bodySubgraphIndex
  var_handles: VarHandles = process_subgraph(context, cond_subgraph_index)
  var_handles |= process_subgraph(context, body_subgraph_index)
  for var_handle_id in var_handles:
    for read_var_op in context.get_pending_read_var_ops_by_handle(
        var_handle_id, op.subgraph.index):
      context.remove_pending_op(read_var_op)
      process_read_var_pending_ops(context, read_var_op)

  process_pending_ops(context)

  return var_handles


def process_read_var_pending_ops(context: Context,
                                 read_var_op: model_facade._Operator) -> None:
  """Process READ_VARIABLE operator against any pending operators.
  Then add the READ_VARIABLE operator to the list of reordered operations.
  """
  for tensor_input in read_var_op.inputs_indices:
    pending_op = context.get_pending_op(tensor_input,
                                        read_var_op.subgraph.index)
    if pending_op is not None:
      context.remove_pending_op(pending_op)
      context.push_current_op(pending_op)
      process_pending_ops(context)
      context.pop_current_op()

  context.append_to_reordered_operations(read_var_op)


def process_pending_ops(context: Context) -> None:
  """Process current operator against any pending operators.
  Then add the current operator to the list of reordered operations.
  """
  op = context.current_op()
  if op.inputs_indices is not None:
    for tensor_input in op.inputs_indices:
      pending_op = context.get_pending_op(tensor_input, op.subgraph.index)
      if pending_op is not None:
        context.remove_pending_op(pending_op)
        context.push_current_op(pending_op)
        process_pending_ops(context)
        context.pop_current_op()

  context.append_to_reordered_operations(op)


def process_subgraph_pending_ops(context: Context,
                                 subgraph_index: SubgraphIndex) -> None:
  """Process subgraph outputs against any pending operators.
  """
  outputs_indices = context.model.subgraphs[subgraph_index].outputs_indices
  if outputs_indices is not None:
    for tensor_index in outputs_indices:
      pending_op = context.get_pending_op(tensor_index, subgraph_index)
      if pending_op is not None:
        context.remove_pending_op(pending_op)
        context.push_current_op(pending_op)
        process_pending_ops(context)
        context.pop_current_op()


def process_operator(context: Context) -> VarHandles:
  op = context.current_op()
  if op.builtin_opcode == tflite.BuiltinOperator.VAR_HANDLE:
    process_operator_var_handle(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.ASSIGN_VARIABLE:
    return process_operator_assign_variable(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.IF:
    return process_operator_if(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.WHILE:
    return process_operator_while(context)
  elif op.builtin_opcode == tflite.BuiltinOperator.CALL_ONCE:
    return process_operator_call_once(context)
  elif context.can_be_pending_op(op):
    context.add_pending_op(op)
  else:
    process_pending_ops(context)

  return set()


def process_subgraph(context: Context, subgraph_index: int) -> VarHandles:
  if context.is_subgraph_processed(subgraph_index):
    return context.subgraph_var_handles(subgraph_index)

  var_handles: VarHandles = set()
  subgraph: model_facade._Subgraph = context.model.subgraphs[subgraph_index]
  op: model_facade._Operator

  for op in subgraph.operators:
    context.push_current_op(op)
    var_handles_processed: VarHandles = process_operator(context)
    var_handles.update(var_handles_processed)
    context.pop_current_op()

  process_subgraph_pending_ops(context, subgraph_index)

  operators: List[tflite.OperatorT] = []
  for op in context.reordered_operations(subgraph_index):
    operators.append(op.operator)
  context.model.root.subgraphs[subgraph_index].operators = operators

  context.mark_subgraph_processed(subgraph_index)
  context.set_subgraph_var_handles(subgraph_index, var_handles)

  return var_handles


def op_names_to_values(op_names: List[str]) -> List[int]:
  op_values = []
  builtin_operators = vars(tflite.BuiltinOperator)
  for name in op_names:
    value = builtin_operators.get(name.upper())
    if value is None:
      raise ValueError(f'unknowm operator: {name}')
    else:
      op_values.append(value)
  return op_values


def main(_):
  input_path = Path(FLAGS.input)
  output_path = Path(FLAGS.output)

  with open(input_path, 'rb') as file:
    buffer = bytes(file.read())
  input_model: model_facade._Model = model_facade.read(buffer)

  context = Context(input_model, op_names_to_values(FLAGS.ops))
  _ = process_subgraph(context, 0)

  output_model: bytearray = input_model.compile()
  with open(output_path, 'wb') as file:
    file.write(output_model)
  model_transforms_utils.tflite_flatbuffer_align(str(output_path),
                                                 str(output_path))
  print("\nreordering and alignment completed.")


if __name__ == '__main__':
  app.run(main)
