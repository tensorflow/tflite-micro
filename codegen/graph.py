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

from typing import List, Sequence

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.tools import visualize


def _to_pascal_case(s: str) -> str:
  return s.title().replace('_', '')


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

  def __init__(self, model: schema_fb.ModelT, operator: schema_fb.OperatorT):
    self._operator: schema_fb.OperatorT = operator
    self._op_code: OpCode = OpCode(
        model.operatorCodes[self._operator.opcodeIndex])

  @property
  def op_code(self) -> OpCode:
    return self._op_code


class Subgraph(object):

  def __init__(self, model: schema_fb.ModelT, subgraph: schema_fb.SubGraphT):
    self._subgraph: schema_fb.SubGraphT = subgraph
    self._operators: List[Operator] = [
        Operator(model, operator) for operator in subgraph.operators
    ]

  @property
  def operators(self) -> Sequence[Operator]:
    return self._operators


class Graph(object):

  def __init__(self, model: schema_fb.ModelT):
    self._subgraphs: List[SubGraph] = [
        Subgraph(model, subgraph) for subgraph in model.subgraphs
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
