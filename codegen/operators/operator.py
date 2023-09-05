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

import abc
from typing import Optional
import string
import textwrap

from tflite_micro.codegen import utils
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb


class Operator(abc.ABC):

  def __init__(self, operator: schema_fb.OperatorT):
    self._operator: schema_fb.OperatorT = operator
    self._inputs: utils.IntArray = utils.IntArray(self._operator.inputs)
    self._outputs: utils.IntArray = utils.IntArray(self._operator.outputs)
    self._intermediates: Optional[utils.IntArray] = utils.IntArray(
        self._operator.intermediates) if self._operator.intermediates else None

  def generate_c_node_data(self, type_name: str, node_name: str) -> str:
    struct_template = string.Template("struct ${type_name} {\n"
                                      "${body}"
                                      "} ${node_name};")
    body_template = string.Template("${inputs}\n"
                                    "${outputs}\n"
                                    "${intermediates}\n"
                                    "${builtin_data}\n")
    if self._intermediates:
      intermediates = self._intermediates.generate_c_struct(
          "Intermediates", "intermediates")
    else:
      intermediates = "// No intermediates"

    body = body_template.substitute(
        inputs=self._inputs.generate_c_struct("Inputs", "inputs"),
        outputs=self._outputs.generate_c_struct("Outputs", "outputs"),
        intermediates=intermediates,
        builtin_data=self.generate_c_builtin_data())

    return struct_template.substitute(type_name=type_name,
                                      node_name=node_name,
                                      body=textwrap.indent(body, "  "))

  def generate_c_node_init(self, tflite_node_name: str,
                           node_data_name: str) -> str:
    init_template = string.Template(
        "${tflite_node_name} = TfLiteNode{\n"
        "    .inputs ="
        " reinterpret_cast<TfLiteIntArray*>(&${node_data_name}.inputs),\n"
        "    .outputs ="
        " reinterpret_cast<TfLiteIntArray*>(&${node_data_name}.outputs),\n"
        "    .intermediates = ${intermediates},\n"
        "    .user_data = nullptr,\n"
        "    .builtin_data ="
        " static_cast<void*>(&${node_data_name}.builtin_data),\n"
        "    .custom_initial_data = nullptr,\n"
        "    .custom_initial_data_size = 0};")

    if self._intermediates:
      intermediates = (
          "reinterpret_cast<TfLiteIntArray*>(&{}.intermediates)".format(
              self._intermediates))
    else:
      intermediates = "nullptr"

    return init_template.substitute(tflite_node_name=tflite_node_name,
                                    node_data_name=node_data_name,
                                    intermediates=intermediates)

  @property
  def op_code_index(self) -> int:
    return self._operator.opcodeIndex

  @abc.abstractmethod
  def generate_c_builtin_data(self) -> str:
    raise NotImplementedError(f"Generating builtin data in {self.__name__}")
