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

from typing import Dict, List, Optional, Sequence
import string
import textwrap

from tflite_micro.codegen.operators import factory
from tflite_micro.codegen.operators import operator
from tflite_micro.codegen import tensor
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

  def __init__(self, model: schema_fb.ModelT, buffers: Sequence[tensor.Buffer],
               subgraph_idx: int, subgraph: schema_fb.SubGraphT):
    self._subgraph_idx: int = subgraph_idx
    self._subgraph: schema_fb.SubGraphT = subgraph
    self._op_codes: List[OpCode] = [
        OpCode(op_code) for op_code in model.operatorCodes
    ]
    self._tensors: List[Tensor] = []
    for t in subgraph.tensors:
      self._tensors.append(tensor.Tensor(buffers[t.buffer], t))

    self._operators: List[operator.Operator] = []
    for op in subgraph.operators:
      op_code = model.operatorCodes[op.opcodeIndex]
      self._operators.append(factory.create_operator(op_code, op))

  @property
  def index(self) -> int:
    return self._subgraph_idx

  @property
  def inputs(self) -> Sequence[int]:
    return self._subgraph.inputs

  @property
  def outputs(self) -> Sequence[int]:
    return self._subgraph.outputs

  @property
  def operators(self) -> Sequence[operator.Operator]:
    return self._operators

  @property
  def tensors(self) -> Sequence[tensor.Tensor]:
    return self._tensors

  @property
  def needs_zero_length_int_array(self) -> bool:
    return any(t.needs_zero_length_int_array for t in self.tensors)

  @property
  def invoke_fn_name(self) -> str:
    return f"InvokeSubgraph{self.index}"

  @property
  def inputs_array_name(self) -> str:
    return f"kSubgraph{self.index}Inputs"

  @property
  def outputs_array_name(self) -> str:
    return f"kSubgraph{self.index}Outputs"

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
    function_template = string.Template(
        "TfLiteStatus ${function_name}(TfLiteContext* context,\n"
        "                             tflite::Span<TfLiteNode> nodes) {\n"
        "  TFLITE_DCHECK(nodes.size() == ${num_nodes});\n"
        "${body}\n"
        "  return kTfLiteOk;\n"
        "}")

    body_template = string.Template(
        "  TF_LITE_ENSURE_OK(\n"
        "      context, op_table[${op_code}].invoke(context, &${node}));\n")
    invoke_strs: List[str] = []
    for op_idx, op in enumerate(self.operators):
      invoke_strs.append(
          body_template.substitute(
              op_code=self._op_codes[op.op_code_index].full_enum_name,
              node=f"nodes[{op_idx}]"))

    invoke = function_template.substitute(function_name=self.invoke_fn_name,
                                          num_nodes=len(self.operators),
                                          body="".join(invoke_strs))
    return textwrap.indent(invoke, indent)

  def generate_c_input_array(self, indent: str) -> str:
    return utils.generate_c_int_array(indent, "size_t", self.inputs_array_name,
                                      self.inputs)

  def generate_c_output_array(self, indent: str) -> str:
    return utils.generate_c_int_array(indent, "size_t",
                                      self.outputs_array_name, self.outputs)

  def generate_c_subgraph_init(self, indent: str) -> str:
    init_template = string.Template(
        "{.inputs = {&${input_array}[0], ${input_size}},\n"
        " .outputs = {&${output_array}[0], ${output_size}},\n"
        " .nodes = {&${node_array}[0], ${node_size}},\n"
        " .tensors = {&${tensor_array}[0], ${tensor_size}},\n"
        " .invoke = &${invoke}},")
    return textwrap.indent(
        init_template.substitute(input_array=self.inputs_array_name,
                                 input_size=len(self.inputs),
                                 output_array=self.outputs_array_name,
                                 output_size=len(self.outputs),
                                 node_array=self.nodes_array,
                                 node_size=len(self.operators),
                                 tensor_array=self.tensors_array,
                                 tensor_size=len(self.tensors),
                                 invoke=self.invoke_fn_name), indent)

  @property
  def tensors_array(self) -> str:
    return f"subgraph{self.index}_tensors_"

  def tensors_element(self, tensor_idx: int) -> str:
    return self.tensors_array + f"[{tensor_idx}]"

  def tensor_data_type(self, tensor_idx: int) -> str:
    return f"Tensor{self.index}_{tensor_idx}"

  def tensor_data_name(self, tensor_idx: int) -> str:
    return f"tensor{self.index}_{tensor_idx}"

  def generate_c_tensor_data(self, indent: str) -> str:
    tensor_dims_strs: List[str] = []
    for tensor_idx, tensor in enumerate(self.tensors):
      type_name = self.tensor_data_type(tensor_idx)
      tensor_name = self.tensor_data_name(tensor_idx)
      tensor_dims_strs.append(
          tensor.generate_c_tensor_dims(type_name, tensor_name))
    return textwrap.indent("\n\n".join(tensor_dims_strs), indent)

  def generate_c_tensor_init(self, indent: str) -> str:
    tensor_init_strs: List[str] = []
    for tensor_idx, tensor in enumerate(self.tensors):
      tflite_tensor_name = self.tensors_element(tensor_idx)
      tensor_data_name = self.tensor_data_name(tensor_idx)
      tensor_init_strs.append(
          tensor.generate_c_tensor_init(tflite_tensor_name, tensor_data_name))
    return textwrap.indent("\n".join(tensor_init_strs), indent)


class Graph(object):

  def __init__(self, model: schema_fb.ModelT):
    buffers: List[tensor.Buffer] = [
        tensor.Buffer("buffer_{}".format(idx), buffer)
        for idx, buffer in enumerate(model.buffers)
    ]
    self._subgraphs: List[SubGraph] = [
        Subgraph(model, buffers, idx, subgraph)
        for idx, subgraph in enumerate(model.subgraphs)
    ]

  @property
  def subgraphs(self) -> Sequence[Subgraph]:
    return self._subgraphs

  @property
  def buffers(self) -> Sequence[tensor.Buffer]:
    buffers: List[tensor.Buffer] = []
    for subgraph in self.subgraphs:
      for t in subgraph.tensors:
        buffers.append(t.buffer)
    return buffers

  @property
  def needs_zero_length_int_array(self) -> bool:
    return any(subgraph.needs_zero_length_int_array
               for subgraph in self.subgraphs)


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
