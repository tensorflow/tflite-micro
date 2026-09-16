# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Operator inputs that must stay constant.

Compression replaces a constant tensor with the output of a DECODE
operator, and a DECODE output is never constant: decode_insert gives it
no buffer, so the allocator places it in the arena, and the values
arrive only when DECODE runs in Invoke. A kernel that needs the data
earlier, in Prepare, therefore cannot read a compressed tensor. It
guards the input with IsConstantTensor and returns an error, so the
whole model fails to prepare.

Listed here are the inputs every implementation requires, which the
portable kernels guard. All are shape or parameter inputs the converter
bakes into the model. Several optimized kernels impose requirements of
their own, on weights among others, and a few read weights in Prepare
with no guard at all. Those are deliberately absent: they hold only for
one target, and refusing them everywhere would rule out compressing
convolution filters for every target that is unaffected.
"""

from dataclasses import dataclass

from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


@dataclass(frozen=True)
class Requirement:
  """A kernel's need for one operator input to hold data in Prepare.

  Attributes:
    input_name: The input's name in the kernel, e.g. "paddings".
    source: The kernel source recording the requirement.
  """
  input_name: str
  source: str


# Keyed by (builtin operator, input position).
_REQUIREMENTS: dict[tuple[int, int], Requirement] = {
    (tflite.BuiltinOperator.PAD, 1):
    Requirement("paddings", "kernels/pad_common.cc"),
    (tflite.BuiltinOperator.PADV2, 1):
    Requirement("paddings", "kernels/pad_common.cc"),
    (tflite.BuiltinOperator.TRANSPOSE, 1):
    Requirement("perm", "kernels/transpose_common.cc"),
    (tflite.BuiltinOperator.STRIDED_SLICE, 1):
    Requirement("begin", "kernels/strided_slice_common.cc"),
    (tflite.BuiltinOperator.STRIDED_SLICE, 2):
    Requirement("end", "kernels/strided_slice_common.cc"),
    (tflite.BuiltinOperator.STRIDED_SLICE, 3):
    Requirement("strides", "kernels/strided_slice_common.cc"),
    (tflite.BuiltinOperator.EXPAND_DIMS, 1):
    Requirement("axis", "kernels/expand_dims.cc"),
    (tflite.BuiltinOperator.FILL, 0):
    Requirement("dims", "kernels/fill.cc"),
    (tflite.BuiltinOperator.BROADCAST_TO, 1):
    Requirement("shape", "kernels/broadcast_to.cc"),
    (tflite.BuiltinOperator.SPLIT, 0):
    Requirement("axis", "kernels/split.cc"),
    (tflite.BuiltinOperator.SPLIT_V, 2):
    Requirement("axis", "kernels/split_v.cc"),
    (tflite.BuiltinOperator.RESIZE_BILINEAR, 1):
    Requirement("size", "kernels/resize_bilinear.cc"),
    (tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR, 1):
    Requirement("size", "kernels/resize_nearest_neighbor.cc"),
}


@dataclass(frozen=True)
class Use:
  """One input position, of one operator kind, needing a tensor constant.

  A constant often feeds the same input of many operators, so a use
  covers every operator sharing an input position and requirement
  rather than naming each separately.

  Attributes:
    operators: Indices of the operators within their subgraph.
    operator_name: Display name of the operator, e.g. "PAD".
    position: The input position the tensor occupies.
    requirement: What the kernel needs and where that is recorded.
  """
  operators: tuple[int, ...]
  operator_name: str
  position: int
  requirement: Requirement

  def describe(self) -> str:
    """Explains the requirement in one line."""
    if len(self.operators) == 1:
      where = f"operator {self.operators[0]}"
    elif len(self.operators) <= 4:
      where = "operators " + ", ".join(str(i) for i in self.operators)
    else:
      where = f"{len(self.operators)} operators"
    return (f"{self.operator_name} rejects a non-constant "
            f"{self.requirement.input_name!r} at input {self.position} "
            f"({where}, {self.requirement.source})")


def find_uses(subgraph: model_editor.Subgraph,
              tensor: model_editor.Tensor) -> list[Use]:
  """Finds the uses of a tensor that need it to stay constant.

  Args:
    subgraph: The subgraph holding the tensor.
    tensor: The tensor to examine.

  Returns:
    A Use for each (operator kind, input position) reading the tensor
    whose kernel needs the data in Prepare. Any use rules the tensor out
    for compression.
  """
  groups: dict[tuple[str, int], list[int]] = {}
  requirements: dict[tuple[str, int], Requirement] = {}
  for operator in subgraph.operators:
    for position, operand in enumerate(operator.inputs):
      if operand is not tensor:
        continue
      requirement = _REQUIREMENTS.get((operator.opcode, position))
      if requirement is None:
        continue
      key = (operator.opcode_name, position)
      groups.setdefault(key, []).append(operator.index)
      requirements[key] = requirement

  return [
      Use(operators=tuple(operators),
          operator_name=name,
          position=position,
          requirement=requirements[(name, position)])
      for (name, position), operators in groups.items()
  ]
