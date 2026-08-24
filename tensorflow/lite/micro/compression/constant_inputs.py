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
operator. That output has no buffer in the model, so it lives in the
arena and holds no values until DECODE runs in Invoke. A kernel that
reads an input in Prepare checks it with IsConstantTensor and returns an
error when the input is not constant, and the model fails to prepare.

The table lists the inputs that the portable kernels check this way.
Each is a shape or parameter that the converter stores as a constant.
Some optimized kernels read more inputs in Prepare, weights among them,
and a few read weights without a check. The table leaves those out.
They apply to one target, and refusing them would block compression of
convolution filters on every other target.
"""

from dataclasses import dataclass

from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


# Input names, keyed by (builtin operator, input position).
_REQUIRED_INPUTS: dict[tuple[int, int], str] = {
  (tflite.BuiltinOperator.PAD, 1): "paddings",
  (tflite.BuiltinOperator.PADV2, 1): "paddings",
  (tflite.BuiltinOperator.TRANSPOSE, 1): "perm",
  (tflite.BuiltinOperator.STRIDED_SLICE, 1): "begin",
  (tflite.BuiltinOperator.STRIDED_SLICE, 2): "end",
  (tflite.BuiltinOperator.STRIDED_SLICE, 3): "strides",
  (tflite.BuiltinOperator.EXPAND_DIMS, 1): "axis",
  (tflite.BuiltinOperator.FILL, 0): "dims",
  (tflite.BuiltinOperator.BROADCAST_TO, 1): "shape",
  (tflite.BuiltinOperator.SPLIT, 0): "axis",
  (tflite.BuiltinOperator.SPLIT_V, 2): "axis",
  (tflite.BuiltinOperator.RESIZE_BILINEAR, 1): "size",
  (tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR, 1): "size",
}


@dataclass(frozen=True)
class Use:
  """A tensor's use as an input that must stay constant.

  One Use covers every operator of the same type that reads the
  tensor at the same input position, e.g. all PAD operators that take
  it as paddings.

  Attributes:
    operators: Indices of the operators within their subgraph.
    operator_name: Display name of the operator, e.g. "PAD".
    position: The input position the tensor occupies.
    input_name: The input's name in the kernel, e.g. "paddings".
  """

  operators: tuple[int, ...]
  operator_name: str
  position: int
  input_name: str

  def describe(self) -> str:
    """Describes the use in one line for an error message."""
    if len(self.operators) == 1:
      where = f"operator {self.operators[0]}"
    elif len(self.operators) <= 4:
      where = "operators " + ", ".join(str(i) for i in self.operators)
    else:
      where = f"{len(self.operators)} operators"
    return (
      f"{self.operator_name} rejects a non-constant "
      f"{self.input_name!r} at input {self.position} ({where})"
    )


def find_uses(
  subgraph: model_editor.Subgraph, tensor: model_editor.Tensor
) -> list[Use]:
  """Finds the operator inputs that need a tensor to stay constant.

  Args:
    subgraph: The subgraph holding the tensor.
    tensor: The tensor to examine.

  Returns:
    A Use for each operator type and input position whose kernel reads
    the tensor in Prepare. A non-empty list means the tensor cannot be
    compressed.
  """
  groups: dict[tuple[str, int], list[int]] = {}
  input_names: dict[tuple[str, int], str] = {}
  for operator in subgraph.operators:
    for position, operand in enumerate(operator.inputs):
      if operand is not tensor:
        continue
      input_name = _REQUIRED_INPUTS.get((operator.opcode, position))
      if input_name is None:
        continue
      key = (operator.opcode_name, position)
      groups.setdefault(key, []).append(operator.index)
      input_names[key] = input_name

  return [
    Use(
      operators=tuple(operators),
      operator_name=name,
      position=position,
      input_name=input_names[(name, position)],
    )
    for (name, position), operators in groups.items()
  ]
