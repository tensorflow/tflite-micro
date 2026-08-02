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
"""Proposes a compression spec from a model.

See USAGE.
"""

import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional, Union

import absl.app
import absl.flags
import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

is_bazel = "BUILD_WORKING_DIRECTORY" in os.environ or "BAZEL_TEST" in os.environ

if is_bazel:
  _COMMAND = "bazel run //tensorflow/lite/micro/compression:propose_spec --"
  _EPILOG = textwrap.dedent("""\


      Note: When running through Bazel, paths must be absolute.

      Example:
        bazel run //tensorflow/lite/micro/compression:propose_spec -- \\
            $(realpath model.tflite) > spec.yaml""")
else:
  _COMMAND = os.path.basename(sys.argv[0])
  _EPILOG = ""

USAGE = textwrap.dedent(f"""\
    Usage: {_COMMAND} \\
        [--norequire_savings] [--output <spec.yaml>] <MODEL_PATH>

    Propose a compression spec for a .tflite model. The proposal lists every
    constant tensor that LUT compression can encode, with the minimum
    index_bitwidth, ready for a human to review and prune. By default,
    list only tensors that compression would shrink; --norequire_savings
    lists every LUT-encodable constant. Output goes to stdout unless
    --output is given.""") + _EPILOG

# LUT compression packs indices in 1 to 7 bits, so a value table may hold at
# most 2**7 entries.
_MAX_BITWIDTH = 7

# Each compressed tensor gains an ancillary buffer holding a 16-byte decode
# header followed by its value tables.
_ANCILLARY_HEADER_BYTES = 16

_OPERATOR_NAMES = {
    code: name
    for name, code in vars(tflite.BuiltinOperator).items()
    if not name.startswith("_")
}

_TYPE_NAMES = {
    value: name
    for name, value in vars(tflite.TensorType).items()
    if not name.startswith("_")
}


@dataclass
class Candidate:
  """A tensor that LUT compression can encode, and its size arithmetic.

  Attributes:
    subgraph: Index of the subgraph holding the tensor.
    tensor: Index of the tensor within its subgraph.
    name: Display name, quoted, or "(unnamed)".
    type_name: Name of the tensor's element type, e.g. "INT32".
    shape: The tensor's dimensions as plain ints.
    elements: Number of elements in the tensor.
    bitwidth: Minimum index_bitwidth able to enumerate the tensor's
        unique values.
    axis: Compression axis, or None for one table over the whole tensor.
    tables: Number of value tables.
    max_unique: Number of unique values in the worst table.
    original_bytes: Size of the tensor's uncompressed buffer.
    estimated_bytes: Estimated compressed size, counting the packed
        indices, the value tables, and the decode header.
    consumers: Descriptions of the tensor's uses as an operator input.
    sharers: (subgraph, tensor) coordinates of the other constants
        referencing the tensor's buffer.
  """

  subgraph: int
  tensor: int
  name: str
  type_name: str
  shape: list[int]
  elements: int
  bitwidth: int
  axis: Optional[int]
  tables: int
  max_unique: int
  original_bytes: int
  estimated_bytes: int
  consumers: list[str]
  sharers: list[tuple[int, int]]

  @property
  def savings(self) -> int:
    """Bytes saved by compression; negative when compression grows it."""
    return self.original_bytes - self.estimated_bytes


@dataclass
class Reject:
  """A constant tensor left out of the proposal, and why.

  Attributes:
    subgraph: Index of the subgraph holding the tensor.
    tensor: Index of the tensor within its subgraph.
    name: Display name, quoted, or "(unnamed)".
    type_name: Name of the tensor's element type, e.g. "INT32".
    shape: The tensor's dimensions as plain ints.
    reason: Why the tensor is left out.
  """

  subgraph: int
  tensor: int
  name: str
  type_name: str
  shape: list[int]
  reason: str


def propose(model_bytes: bytes,
            model_name: str = "model",
            require_savings: bool = True) -> str:
  """Returns a commented YAML compression spec proposed from the model.

  Args:
    model_bytes: A .tflite flatbuffer.
    model_name: A name for the model, used only in the header comment.
    require_savings: If true, list only tensors that compression would
        shrink. If false, list every LUT-encodable constant.
  """
  model = model_editor.read(model_bytes)
  candidates, rejects = survey(model, require_savings=require_savings)
  return _render(candidates, rejects, model_name, model_size=len(model_bytes))


def survey(
    model: model_editor.Model,
    require_savings: bool = True) -> tuple[list[Candidate], list[Reject]]:
  """Walks the model and splits its constants into candidates and rejects.

  Args:
    model: The model to survey.
    require_savings: If true, move candidates whose estimated compressed
        size does not beat the original into the rejects.

  Returns:
    A (candidates, rejects) tuple, each a list in model order.
  """
  buffer_users = {}
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.buffer is not None and len(tensor.buffer.data) > 0:
        buffer_users.setdefault(id(tensor.buffer), []).append(
            (subgraph.index, tensor.index))

  candidates = []
  rejects = []
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      result = _analyze(subgraph, tensor, buffer_users)
      if result is None:
        continue
      if isinstance(result, Candidate) and require_savings \
          and result.savings <= 0:
        result = Reject(subgraph=result.subgraph,
                        tensor=result.tensor,
                        name=result.name,
                        type_name=result.type_name,
                        shape=result.shape,
                        reason=f"no savings ({result.original_bytes:,} -> "
                        f"{result.estimated_bytes:,} bytes)")
      if isinstance(result, Candidate):
        candidates.append(result)
      else:
        rejects.append(result)
  return candidates, rejects


def _analyze(
    subgraph: model_editor.Subgraph, tensor: model_editor.Tensor,
    buffer_users: dict[int, list[tuple[int, int]]]
) -> Union[Candidate, Reject, None]:
  """Sizes up one tensor for LUT compression.

  Args:
    subgraph: The subgraph holding the tensor.
    tensor: The tensor to analyze.
    buffer_users: Map of buffer object id to the (subgraph, tensor)
        coordinates of every constant referencing that buffer.

  Returns:
    A Candidate, a Reject explaining why the tensor cannot be
    LUT-compressed, or None if the tensor is not a constant.
  """
  if tensor.buffer is None or len(tensor.buffer.data) == 0:
    return None

  # Coerce possible numpy scalars so the dimensions render as plain ints.
  shape = [int(d) for d in tensor.shape]

  def reject(reason):
    return Reject(subgraph=subgraph.index,
                  tensor=tensor.index,
                  name=_display_name(tensor),
                  type_name=_TYPE_NAMES.get(tensor.dtype, str(tensor.dtype)),
                  shape=shape,
                  reason=reason)

  # Note the tensors aliasing this tensor's buffer, where the converter
  # deduplicated identical constants. The compressor handles aliases, but
  # only all together: a spec covering some aliases of a buffer and not
  # others compresses none of them.
  sharers = [
      coords for coords in buffer_users[id(tensor.buffer)]
      if coords != (subgraph.index, tensor.index)
  ]

  try:
    array = tensor.array
  except ValueError as e:
    return reject(str(e))

  try:
    axis = lut.identify_compression_axis(tensor)
  except compressor.CompressionError as e:
    return reject(str(e))

  tables, max_unique = _count_unique(array, axis)
  if max_unique > 2**_MAX_BITWIDTH:
    return reject(f"{max_unique:,} unique values exceed "
                  f"a {_MAX_BITWIDTH}-bit LUT index")

  bitwidth = (max_unique - 1).bit_length() or 1
  indices_bytes = (array.size * bitwidth + 7) // 8
  ancillary_bytes = (_ANCILLARY_HEADER_BYTES +
                     tables * max_unique * array.itemsize)

  return Candidate(subgraph=subgraph.index,
                   tensor=tensor.index,
                   name=_display_name(tensor),
                   type_name=_TYPE_NAMES.get(tensor.dtype, str(tensor.dtype)),
                   shape=shape,
                   elements=array.size,
                   bitwidth=bitwidth,
                   axis=axis,
                   tables=tables,
                   max_unique=max_unique,
                   original_bytes=len(tensor.buffer.data),
                   estimated_bytes=indices_bytes + ancillary_bytes,
                   consumers=_describe_consumers(subgraph, tensor),
                   sharers=sharers)


def _display_name(tensor: model_editor.Tensor) -> str:
  return f'"{tensor.name}"' if tensor.name else "(unnamed)"


def _count_unique(array: np.ndarray, axis: Optional[int]) -> tuple[int, int]:
  """Returns (tables, max unique values per table) for the given axis.

  Axis None means one table for the whole tensor. Otherwise the count
  mirrors the compressor, which builds one table per slice along the
  axis, so the bitwidth must cover the worst slice.
  """
  if axis is None:
    return 1, len(np.unique(array))
  slices = np.moveaxis(array, axis, 0)
  return len(slices), max(len(np.unique(s)) for s in slices)


def _describe_consumers(subgraph: model_editor.Subgraph,
                        tensor: model_editor.Tensor) -> list[str]:
  """Describes the uses of the tensor as an operator input.

  One description covers each (input position, operator name) group,
  summarizing a large group by its count, so a constant shared by
  dozens of operators still yields a readable comment line.
  """
  groups = {}
  for op in subgraph.operators:
    if op.custom_code is not None:
      name = op.custom_code
    else:
      name = _OPERATOR_NAMES.get(op.opcode, f"opcode {op.opcode}")
    for position, t in enumerate(op.inputs):
      if t is tensor:
        groups.setdefault((position, name), []).append(op.index)

  descriptions = []
  for (position, name), op_indices in groups.items():
    if len(op_indices) == 1:
      where = f"operator {op_indices[0]}"
    elif len(op_indices) <= 4:
      where = "operators " + ", ".join(str(i) for i in op_indices)
    else:
      where = f"{len(op_indices)} operators"
    descriptions.append(f"input {position} of {name} ({where})")
  return descriptions or ["not an operator input"]


def _render(candidates: list[Candidate], rejects: list[Reject],
            model_name: str, model_size: int) -> str:
  """Renders the survey results as a commented YAML spec."""
  lines = []
  lines.append(f"# Compression spec proposed for {model_name}.")
  lines.append("#")
  lines.append("# Each entry names a constant tensor that LUT compression "
               "can encode and")
  lines.append("# the minimum index_bitwidth able to enumerate the tensor's "
               "unique values.")
  lines.append("# Sizes count the packed indices, the value tables, and the "
               "decode")
  lines.append("# header. Review the entries and delete those for tensors "
               "that should")
  lines.append("# stay uncompressed.")
  lines.append("#")
  lines.append("# Tensor, operator, and buffer numbers refer to the input "
               "model.")
  lines.append("# Compression inserts DECODE operators and rewrites tensors "
               "and buffers,")
  lines.append("# so numbers in the compressed model differ.")

  if candidates:
    total_original = sum(c.original_bytes for c in candidates)
    total_estimated = sum(c.estimated_bytes for c in candidates)
    plural = "s" if len(candidates) != 1 else ""
    lines.append("#")
    lines.append(f"# {len(candidates)} tensor{plural}, {total_original:,} -> "
                 f"{total_estimated:,} bytes "
                 f"({_change(total_original, total_estimated)})")
    projected = model_size - (total_original - total_estimated)
    lines.append(f"# whole model, {model_size:,} -> {projected:,} bytes "
                 f"estimated ({_change(model_size, projected)})")

  lines.append("")
  lines.append("tensors:")

  for c in candidates:
    lines.append("")
    lines.append(f"  # {c.name} {c.type_name} {c.shape}, "
                 f"{', '.join(c.consumers)}")
    if c.axis is None:
      uniques = f"{c.max_unique} unique values"
    else:
      uniques = (f"max {c.max_unique} unique values per table, "
                 f"{c.tables} tables along axis {c.axis}")
    lines.append(f"  # {c.elements:,} elements, {uniques}, "
                 f"{c.original_bytes:,} -> {c.estimated_bytes:,} bytes "
                 f"({_change(c.original_bytes, c.estimated_bytes)})")
    if c.sharers:
      others = ", ".join(f"subgraph {s} tensor {t}" for s, t in c.sharers)
      lines.append(f"  # shares a buffer with {others}; keep or delete "
                   "all aliases together")
    lines.append(f"  - subgraph: {c.subgraph}")
    lines.append(f"    tensor: {c.tensor}")
    lines.append("    compression:")
    lines.append("      - lut:")
    lines.append(f"          index_bitwidth: {c.bitwidth}")

  if not candidates:
    lines.append("  []")

  if rejects:
    lines.append("")
    lines.append("# Constant tensors not listed:")
    for r in rejects:
      lines.append(f"#   subgraph {r.subgraph} tensor {r.tensor} "
                   f"{r.name} {r.type_name} {r.shape}: {r.reason}")

  lines.append("")
  return "\n".join(lines)


def _change(original: int, estimated: int) -> str:
  """Describes a size change in bytes and percent, e.g. "16 saved, 50%"."""
  saved = original - estimated
  percent = round(100 * saved / original)
  if saved >= 0:
    return f"{saved:,} saved, {percent}%"
  return f"{-saved:,} larger, {-percent}%"


FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("output", None,
                         "write the spec here instead of stdout")
absl.flags.DEFINE_bool(
    "require_savings", True,
    "list only tensors that compression would shrink; --norequire_savings "
    "lists every LUT-encodable constant")


def main(argv):
  try:
    model_path = argv[1]
  except IndexError:
    sys.stderr.write(USAGE + "\n")
    return 1

  with open(model_path, "rb") as file:
    model_bytes = file.read()

  text = propose(model_bytes,
                 model_name=os.path.basename(model_path),
                 require_savings=FLAGS.require_savings)

  if FLAGS.output:
    with open(FLAGS.output, "w") as file:
      file.write(text)
  else:
    sys.stdout.write(text)

  return 0


if __name__ == "__main__":
  sys.modules["__main__"].__doc__ = USAGE  # for absl's use
  absl.app.run(main)
