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

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.tools import visualize


def _to_pascal_case(s: str) -> str:
  return s.title().replace('_', '')


class IntArray(object):

  def __init__(self, data: List[int]):
    self._data = data

  def c_struct(self, type_name: str, variable_name: Optional[str]) -> str:
    struct_template = string.Template("struct ${type_name} {\n"
                                      "  int size = ${size};\n"
                                      "  int data[${size}] = {${data}};\n"
                                      "}")
    # TODO(rjascani): Make this pretty print in multi-line chunks
    int_strs = ['{}'.format(i) for i in self._data]
    c_struct_str = struct_template.substitute(type_name=type_name,
                                              size=len(int_strs),
                                              data=', '.join(int_strs))
    if variable_name:
      return c_struct_str + " {};".format(variable_name)
    return c_struct_str + ";"


class OpCode(object):

  def __init__(self, op_code: schema_fb.OperatorCodeT):
    self._op_code: schema_fb.OperatorCodeT = op_code

  def name(self) -> str:
    if self._op_code.customCode:
      return self._op_code.customCode
    return visualize.BuiltinCodeToName(self._op_code.builtinCode)

  def register_function(self) -> str:
    return "tflite::RegisterInference_{}".format(self.name())

  def enum_name(self) -> str:
    return "k{}".format(_to_pascal_case(self.name()))


class Operator(object):

  def __init__(self, model: schema_fb.ModelT, subgraph_idx: int,
               operator_idx: int, operator: schema_fb.OperatorT):
    self._operator: schema_fb.OperatorT = operator
    self._subgraph_idx: int = subgraph_idx
    self._operator_idx: int = operator_idx
    self._op_code: OpCode = OpCode(
        model.operatorCodes[self._operator.opcodeIndex])
    self._inputs: IntArray = IntArray(self._operator.inputs)
    self._outputs: IntArray = IntArray(self._operator.outputs)
    self._intermediates: Optional[IntArray] = IntArray(
        self._operator.intermediates) if self._operator.intermediates else None

  @property
  def node_data_c_struct(self) -> str:
    struct_template = string.Template("struct ${type_name} {\n"
                                      "${body}"
                                      "} const ${node_name};")
    body_template = string.Template("${inputs}\n"
                                    "${outputs}\n"
                                    "${intermediates}\n")
    if self._intermediates:
      intermediates = self._intermediates.c_struct("Intermediates",
                                                   "intermediates")
    else:
      intermediates = "// No intermediates"

    body = body_template.substitute(
        inputs=self._inputs.c_struct("Inputs", "inputs"),
        outputs=self._outputs.c_struct("Outputs", "outputs"),
        intermediates=intermediates)

    c_struct_str = struct_template.substitute(
        type_name=self.node_data_type_name,
        node_name=self.node_data_variable_name,
        body=textwrap.indent(body, "  "))
    return c_struct_str

  @property
  def node_data_type_name(self) -> str:
    return "Node{}_{}".format(self._subgraph_idx, self._operator_idx)

  @property
  def node_data_variable_name(self) -> str:
    return "node_{}_{}".format(self._subgraph_idx, self._operator_idx)

  @property
  def node_element(self) -> str:
    return "subgraph{}_nodes_[{}]".format(self._subgraph_idx,
                                          self._operator_idx)

  @property
  def node_data_inputs(self) -> str:
    return self.node_data_variable_name + ".inputs"

  @property
  def node_data_outputs(self) -> str:
    return self.node_data_variable_name + ".outputs"

  @property
  def node_data_intermediates(self) -> Optional[str]:
    return self.node_data_variable_name + ".intermediates" if self._intermediates else None

  @property
  def op_code(self) -> OpCode:
    return self._op_code


class Subgraph(object):

  def __init__(self, model: schema_fb.ModelT, subgraph_idx: int,
               subgraph: schema_fb.SubGraphT):
    self._subgraph_idx: int = subgraph_idx
    self._subgraph: schema_fb.SubGraphT = subgraph
    self._operators: List[Operator] = [
        Operator(model, self._subgraph_idx, idx, operator)
        for idx, operator in enumerate(subgraph.operators)
    ]

  @property
  def index(self) -> int:
    return self._subgraph_idx

  @property
  def operators(self) -> Sequence[Operator]:
    return self._operators


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
