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
# ==============================================================================
""" Provides object representation for the model that is conducive to code 
    generation using templates. """

from typing import List, Optional, Sequence
import string
import textwrap

from tflite_micro.codegen.operators import operator
from tflite_micro.codegen import utils
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.tools import visualize


class OpCode(object):

  def __init__(self, op_code: schema_fb.OperatorCodeT):
    self._op_code: schema_fb.OperatorCodeT = op_code

  @property
  def name(self) -> str:
    if self._op_code.customCode:
      return self._op_code.customCode
    return visualize.BuiltinCodeToName(self._op_code.builtinCode)

  @property
  def register_function(self) -> str:
    return "tflite::RegisterInference_{}".format(self.name)

  @property
  def enum_name(self) -> str:
    return "k{}".format(utils.to_pascal_case(self.name))

  @property
  def full_enum_name(self) -> str:
    return "OpCode::" + self.enum_name


class Subgraph(object):

  def __init__(self, model: schema_fb.ModelT, subgraph_idx: int,
               subgraph: schema_fb.SubGraphT):
    self._subgraph_idx: int = subgraph_idx
    self._subgraph: schema_fb.SubGraphT = subgraph
    self._operators: List[operator.Operator] = [
        operator.Operator(op) for op in subgraph.operators
    ]
    self._op_codes: List[OpCode] = [
        OpCode(op_code) for op_code in model.operatorCodes
    ]

  @property
  def index(self) -> int:
    return self._subgraph_idx

  @property
  def operators(self) -> Sequence[operator.Operator]:
    return self._operators

  @property
  def nodes_array(self) -> str:
    return f"subgraph{self.index}_nodes_"

  def nodes_element(self, operator_idx: int) -> str:
    return self.nodes_array + f"[{operator_idx}]"

  def node_data_type(self, operator_idx: int) -> str:
    return f"Node{self.index}_{operator_idx}"

  def node_data_name(self, operator_idx: int) -> str:
    return f"node_{self.index}_{operator_idx}"

  def generate_c_node_data(self, indent: str) -> str:
    node_data_strs: List[str] = []
    for op_idx, op in enumerate(self.operators):
      type_name = self.node_data_type(op_idx)
      node_name = self.node_data_name(op_idx)
      node_data_strs.append(op.generate_c_node_data(type_name, node_name))
    return textwrap.indent("\n\n".join(node_data_strs), indent)

  def generate_c_node_init(self, indent: str) -> str:
    node_init_strs: List[str] = []
    for op_idx, op in enumerate(self.operators):
      tflite_node_name = self.nodes_element(op_idx)
      node_data_name = self.node_data_name(op_idx)
      node_init_strs.append(
          op.generate_c_node_init(tflite_node_name, node_data_name))
    return textwrap.indent("\n".join(node_init_strs), indent)

  def generate_c_invoke(self, indent: str) -> str:
    invoke_template = string.Template(
        "TF_LITE_ENSURE_OK(context_, op_table[${op_code}].invoke(\n"
        "                                &context_, &${node}));\n")
    invoke_strs: List[str] = []
    for op_idx, op in enumerate(self.operators):
      invoke_strs.append(
          invoke_template.substitute(
              op_code=self._op_codes[op.op_code_index].full_enum_name,
              node=self.nodes_element(op_idx)))
    return textwrap.indent("".join(invoke_strs), indent)


class Graph(object):

  def __init__(self, model: schema_fb.ModelT):
    self._subgraphs: List[SubGraph] = [
        Subgraph(model, idx, subgraph)
        for idx, subgraph in enumerate(model.subgraphs)
    ]

  @property
  def subgraphs(self) -> Sequence[Subgraph]:
    return self._subgraphs


class OpCodeTable(object):

  def __init__(self, models: Sequence[schema_fb.ModelT]):
    op_codes = []
    for model in models:
      for op_code in model.operatorCodes:
        op_codes.append(OpCode(op_code))

    self._op_codes: List([OpCode]) = list(set(op_codes))

  @property
  def op_codes(self) -> Sequence[OpCode]:
    return self._op_codes
