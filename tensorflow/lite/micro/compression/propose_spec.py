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

This tool reads a .tflite model and writes a proposed compression spec
with an entry for each constant tensor that LUT compression can encode.
Each entry uses the per-tensor or per-channel layout that gives the
smallest encoding of the tensor's values. A comment on each entry
identifies the tensor and gives its size before and after, and notes
any other tensors that share its buffer, since those should be kept or
deleted together. A footer lists any tensors left out. A header line
gives the size of the model compressed as proposed. The proposed spec
is a starting point to review and prune.

See USAGE for the command line.
"""

import argparse
import json
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import constant_inputs
from tflite_micro.tensorflow.lite.micro.compression import decode
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import tensor_type
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
  _EPILOG = None

# argparse prints "usage: " ahead of this line.
USAGE = f"""\
{_COMMAND} \\
    [--min_savings <bytes>] [--output <spec.yaml>] <MODEL_PATH>"""

_DESCRIPTION = textwrap.dedent("""\
    Propose a compression spec for a .tflite model. The proposal lists every
    constant tensor that LUT compression can encode, with the mode and
    index_bitwidth that give the smallest encoding, found by testing a
    per-tensor and a per-channel layout against the tensor's values. Review
    the entries and prune them. List only tensors whose compression saves at
    least --min_savings bytes after the DECODE operators and tensors it adds;
    0 lists every LUT-encodable constant. Output goes to stdout unless
    --output is given.""")


def main(argv):
  parser = argparse.ArgumentParser(
    usage=USAGE,
    description=_DESCRIPTION,
    epilog=_EPILOG,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("model_path")
  parser.add_argument(
    "--output", default=None, help="write the spec here instead of stdout"
  )
  parser.add_argument(
    "--min_savings",
    type=_non_negative_int,
    default=_DEFAULT_MIN_SAVINGS,
    help="list only tensors whose compression saves at least this many "
    "bytes after its DECODE overhead; 0 lists every LUT-encodable constant",
  )
  args = parser.parse_args(argv[1:])

  with open(args.model_path, "rb") as file:
    model_bytes = file.read()

  text = propose(
    model_bytes,
    model_name=os.path.basename(args.model_path),
    min_savings=args.min_savings,
  )

  if args.output:
    with open(args.output, "w") as file:
      file.write(text)
  else:
    sys.stdout.write(text)

  return 0


def _non_negative_int(text: str) -> int:
  value = int(text)
  if value < 0:
    raise argparse.ArgumentTypeError("must be a non-negative integer")
  return value


# Default floor on the bytes an entry must save, after the DECODE
# operators and tensors it adds, to be worth proposing. Each DECODE also
# costs time at every inference, so a small net saving is not worth it.
_DEFAULT_MIN_SAVINGS = 512


@dataclass
class Encoding:
  """One way to LUT-encode a tensor, and its size.

  Attributes:
    axis: Compression axis, or None for one table over the whole tensor.
    tables: Number of value tables.
    max_unique: Number of unique values in the largest table.
    bitwidth: Smallest index_bitwidth that can index the largest table.
    estimated_bytes: Packed indices, value tables, and decode header.
  """

  axis: Optional[int]
  tables: int
  max_unique: int
  bitwidth: int
  estimated_bytes: int

  @property
  def indexable(self) -> bool:
    """True when a LUT index can address every value in the largest table."""
    return self.max_unique <= 2**lut.LutAncillaryData.MAX_BITWIDTH


@dataclass
class _Constant:
  """A constant tensor, identified for the proposal's comments.

  Attributes:
    subgraph: Index of the subgraph holding the tensor.
    tensor: Index of the tensor within its subgraph.
    name: Display name, quoted, or "(unnamed)".
    type_name: Name of the tensor's element type, e.g. "INT32".
    shape: The tensor's dimensions as plain ints.
  """

  subgraph: int
  tensor: int
  name: str
  type_name: str
  shape: list[int]


@dataclass
class Candidate(_Constant):
  """A tensor that LUT compression can encode, and its size arithmetic.

  Attributes:
    elements: Number of elements in the tensor.
    encoding: The layout that gives the smallest encoding.
    original_bytes: Size of the tensor's uncompressed buffer.
    consumers: Descriptions of the tensor's readers: operator inputs,
        and the subgraph's outputs.
    sharers: (subgraph, tensor) coordinates of the other constants
        referencing the tensor's buffer.
    net_savings: Bytes that compressing the tensor alone saves in the
        model file, after the DECODE operators and tensors it adds.
  """

  elements: int
  encoding: Encoding
  original_bytes: int
  consumers: list[str]
  sharers: list[tuple[int, int]]
  net_savings: int

  @property
  def estimated_bytes(self) -> int:
    """Estimated compressed size of the tensor."""
    return self.encoding.estimated_bytes

  @property
  def savings(self) -> int:
    """Bytes that compression saves, negative when it enlarges the tensor."""
    return self.original_bytes - self.estimated_bytes

  @property
  def overhead(self) -> int:
    """Bytes that compression adds beside the encoded data."""
    return self.savings - self.net_savings

  def spec_entry(self) -> spec.Tensor:
    """Returns the spec entry that compresses the tensor as proposed."""
    return _spec_entry(self.subgraph, self.tensor, self.encoding)

  @property
  def aliases(self) -> frozenset[tuple[int, int]]:
    """Coordinates of every tensor referencing the buffer, this one included."""
    return frozenset([(self.subgraph, self.tensor), *self.sharers])


@dataclass
class Reject(_Constant):
  """A constant tensor left out of the proposal, and why.

  Attributes:
    reason: Why the tensor is left out.
  """

  reason: str

  @classmethod
  def from_constant(cls, constant: _Constant, reason: str) -> "Reject":
    """Returns a Reject for the constant, giving the reason."""
    identity = {f.name: getattr(constant, f.name) for f in fields(_Constant)}
    return cls(**identity, reason=reason)


def propose(
  model_bytes: bytes,
  model_name: str = "model",
  min_savings: int = _DEFAULT_MIN_SAVINGS,
) -> str:
  """Returns a commented YAML compression spec proposed from the model.

  Args:
    model_bytes: A .tflite flatbuffer.
    model_name: A name for the model, used only in the header comment.
    min_savings: Bytes an entry must save, after the DECODE operators and
        tensors it adds, to be listed. Zero lists every LUT-encodable
        constant.
  """
  model = model_editor.read(model_bytes)
  candidates, rejects = survey(model, min_savings=min_savings)
  compressed_size = None
  if candidates:
    entries = [c.spec_entry() for c in candidates]
    compressed_size = len(_compress(model_bytes, entries))
  return _render(
    candidates,
    rejects,
    model_name,
    model_size=len(model_bytes),
    compressed_size=compressed_size,
  )


def survey(
  model: model_editor.Model, min_savings: int = _DEFAULT_MIN_SAVINGS
) -> tuple[list[Candidate], list[Reject]]:
  """Walks the model and splits its constants into candidates and rejects.

  Args:
    model: The model to survey.
    min_savings: Bytes an entry must save, after the DECODE operators and
        tensors it adds, to stay a candidate. Candidates saving less
        move to the rejects. Zero disables the check, keeping every
        LUT-encodable constant, including those compression grows.

  Returns:
    A (candidates, rejects) tuple, each a list in model order.
  """
  buffer_users = {}
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.buffer is not None and len(tensor.buffer.data) > 0:
        buffer_users.setdefault(id(tensor.buffer), []).append(
          (subgraph.index, tensor.index)
        )

  candidates = []
  rejects = []
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      result = _analyze(subgraph, tensor, buffer_users)
      match result:
        case None:
          continue
        case Candidate() if (
          min_savings > 0 and result.net_savings < min_savings
        ):
          rejects.append(
            Reject.from_constant(result, _shortfall(result, min_savings))
          )
        case Candidate():
          candidates.append(result)
        case Reject():
          rejects.append(result)
  return _drop_partial_aliases(candidates, rejects)


def _analyze(
  subgraph: model_editor.Subgraph,
  tensor: model_editor.Tensor,
  buffer_users: dict[int, list[tuple[int, int]]],
) -> Union[Candidate, Reject, None]:
  """Tests one tensor as a candidate for LUT compression.

  Args:
    subgraph: The subgraph holding the tensor.
    tensor: The tensor to analyze.
    buffer_users: Map of buffer object id to the (subgraph, tensor)
        coordinates of every tensor with data referencing that buffer.

  Returns:
    A Candidate, a Reject explaining why the tensor cannot be
    LUT-compressed, or None if the tensor is not a constant.
  """
  if tensor.buffer is None or len(tensor.buffer.data) == 0:
    return None
  if tensor.is_variable:
    return None

  constant = _Constant(
    subgraph=subgraph.index,
    tensor=tensor.index,
    name=_display_name(tensor),
    type_name=tensor_type.name(tensor.dtype),
    # Coerce possible numpy scalars so the dimensions render as plain ints.
    shape=[int(d) for d in tensor.shape],
  )

  # Other tensors sharing this buffer, where the converter deduplicated
  # identical constants.
  sharers = [
    coords
    for coords in buffer_users[id(tensor.buffer)]
    if coords != (subgraph.index, tensor.index)
  ]

  readers = subgraph.consumers_of(tensor)
  is_output = tensor in subgraph.outputs

  # Cannot compress a tensor that nothing reads, since a DECODE operator
  # goes only before a reader.
  if not readers and not is_output:
    return Reject.from_constant(
      constant, "no operator reads it and it is not a subgraph output"
    )

  uses = constant_inputs.find_uses(subgraph, tensor)
  if uses:
    return Reject.from_constant(
      constant,
      "must stay constant; " + "; ".join(use.describe() for use in uses),
    )

  # Cannot compress a tensor whose values cannot be read as a numpy array,
  # such as a STRING or INT4 tensor.
  try:
    array = tensor.array
  except ValueError as e:
    return Reject.from_constant(constant, str(e))

  # Choose the layout from the values, not the quantization.
  encodings = [_encode(array, axis) for axis in _candidate_axes(array.shape)]
  usable = [e for e in encodings if e.indexable]
  if not usable:
    return Reject.from_constant(
      constant, _overflow_reason(min(encodings, key=lambda e: e.max_unique))
    )

  # _candidate_axes lists the whole-tensor layout first, so a tie goes to
  # the simpler proposal.
  best = min(usable, key=lambda e: e.estimated_bytes)

  return Candidate(
    **vars(constant),
    elements=array.size,
    encoding=best,
    original_bytes=len(tensor.buffer.data),
    consumers=_describe_consumers(subgraph, tensor),
    sharers=sharers,
    net_savings=_measure_net_savings(tensor, len(readers), is_output, best),
  )


def _display_name(tensor: model_editor.Tensor) -> str:
  # JSON quoting escapes a newline or other control character, which would
  # otherwise end the comment line and leave the rest of the name as YAML.
  return (
    json.dumps(tensor.name, ensure_ascii=False) if tensor.name else "(unnamed)"
  )


def _candidate_axes(shape: tuple[int, ...]) -> list[Optional[int]]:
  """Returns the layouts to test, per-tensor first.

  None means one value table for the whole tensor. The kernels decode a
  channel axis only at axis 0 or the last axis. An axis whose slices
  hold one element each is left out, because each element would need
  its own table entry and the indices would add size without saving any.
  """
  axes: list[Optional[int]] = [None]
  rank = len(shape)
  size = int(np.prod(shape)) if rank else 1
  for axis in (0, rank - 1):
    if not 0 <= axis < rank or axis in axes:
      continue
    if size // shape[axis] < 2:
      continue
    axes.append(axis)
  return axes


def _encode(array: np.ndarray, axis: Optional[int]) -> Encoding:
  """Estimates the size of the array LUT-compressed along the given axis."""
  tables, max_unique = _count_unique(array, axis)
  bitwidth = (max_unique - 1).bit_length() or 1
  indices_bytes = (array.size * bitwidth + 7) // 8
  # The ancillary buffer holds a DCM header, then every value table padded
  # to the length of the longest.
  ancillary_bytes = (
    decode.DecodeCommonMetadata.SIZE + tables * max_unique * array.itemsize
  )
  return Encoding(
    axis=axis,
    tables=tables,
    max_unique=max_unique,
    bitwidth=bitwidth,
    estimated_bytes=indices_bytes + ancillary_bytes,
  )


def _count_unique(array: np.ndarray, axis: Optional[int]) -> tuple[int, int]:
  """Returns (tables, unique values in the largest table) for the axis.

  Axis None means one table for the whole tensor. Otherwise there is one
  table per slice along the axis, as in lut.compress_array.
  """
  if axis is None:
    return 1, len(np.unique(array))
  slices = np.moveaxis(array, axis, 0)
  return len(slices), max(len(np.unique(s)) for s in slices)


def _overflow_reason(encoding: Encoding) -> str:
  """Explains that the closest layout still overflows a LUT index."""
  where = (
    "" if encoding.axis is None else f" per channel along axis {encoding.axis}"
  )
  return (
    f"{encoding.max_unique:,} unique values{where} exceed "
    f"a {lut.LutAncillaryData.MAX_BITWIDTH}-bit LUT index"
  )


def _describe_consumers(
  subgraph: model_editor.Subgraph, tensor: model_editor.Tensor
) -> list[str]:
  """Describes the readers of the tensor: operators and subgraph outputs.

  One description covers each (input position, operator name) group,
  summarizing a large group by its count, so a constant shared by
  dozens of operators still yields a readable comment line.
  """
  groups = {}
  for op in subgraph.operators:
    for position, t in enumerate(op.inputs):
      if t is tensor:
        groups.setdefault((position, op.opcode_name), []).append(op.index)

  descriptions = [
    f"input {position} of {name} "
    f"({model_editor.describe_operators(op_indices)})"
    for (position, name), op_indices in groups.items()
  ]
  if tensor in subgraph.outputs:
    descriptions.append("subgraph output")
  return descriptions


def _measure_net_savings(
  tensor: model_editor.Tensor,
  readers: int,
  is_output: bool,
  encoding: Encoding,
) -> int:
  """Measures what compressing the tensor alone saves in a model file.

  Compresses a copy of the tensor in a model of its own, with as many
  reading operators as the original and the same place among the
  subgraph outputs, so the result counts the DECODE operators, the
  decoded and ancillary tensors, and their names and quantization.
  """
  model = model_editor.Model()
  subgraph = model.add_subgraph()
  probe = tensor.copy()
  probe.buffer = model_editor.Buffer(data=tensor.buffer.data)
  subgraph.tensors.append(probe)
  # ABS stands in for any reader; its input need not stay constant.
  for _ in range(readers):
    output = subgraph.add_tensor(shape=tensor.shape, dtype=tensor.dtype)
    subgraph.add_operator(
      opcode=tflite.BuiltinOperator.ABS, inputs=[probe], outputs=[output]
    )
  if is_output:
    subgraph.outputs = [probe]
  model_bytes = bytes(model.build())
  entry = _spec_entry(0, 0, encoding)
  return len(model_bytes) - len(_compress(model_bytes, [entry]))


def _spec_entry(subgraph: int, tensor: int, encoding: Encoding) -> spec.Tensor:
  """Returns the spec entry that compresses the tensor with the encoding."""
  mode = (
    spec.PerTensor()
    if encoding.axis is None
    else spec.PerChannel(axis=encoding.axis)
  )
  return spec.Tensor(
    subgraph=subgraph,
    tensor=tensor,
    compression=[
      spec.LookUpTableCompression(index_bitwidth=encoding.bitwidth, mode=mode)
    ],
  )


def _compress(model_bytes: bytes, entries: list[spec.Tensor]) -> bytes:
  """Compresses the model, without the warning for an entry that grows."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return bytes(compress.compress(model_bytes, entries))


def _shortfall(candidate: Candidate, min_savings: int) -> str:
  """Explains that a candidate does not save enough to be worth listing."""
  sizes = (
    f"{candidate.original_bytes:,} -> {candidate.estimated_bytes:,} bytes, "
    f"plus {candidate.overhead:,} for DECODE"
  )
  if candidate.net_savings <= 0:
    return f"no savings ({sizes})"
  return (
    f"saves {candidate.net_savings:,} bytes ({sizes}), "
    f"under the {min_savings:,} byte floor"
  )


def _drop_partial_aliases(
  candidates: list[Candidate], rejects: list[Reject]
) -> tuple[list[Candidate], list[Reject]]:
  """Rejects each candidate that shares its buffer with an unlisted tensor.

  The compressor compresses the tensors sharing a buffer only all
  together. A tensor left out keeps the original data in the model, so
  compressing its aliases would only enlarge the model.
  """
  listed = {(c.subgraph, c.tensor) for c in candidates}
  kept = []
  for c in candidates:
    missing = [s for s in c.sharers if s not in listed]
    if missing:
      others = ", ".join(f"subgraph {s} tensor {t}" for s, t in missing)
      rejects.append(
        Reject.from_constant(
          c, f"shares a buffer with {others}, which is not listed"
        )
      )
    else:
      kept.append(c)
  rejects.sort(key=lambda r: (r.subgraph, r.tensor))
  return kept, rejects


def _render(
  candidates: list[Candidate],
  rejects: list[Reject],
  model_name: str,
  model_size: int,
  compressed_size: Optional[int],
) -> str:
  """Renders the survey results as a commented YAML spec."""
  lines = textwrap.dedent(f"""\
      # Compression spec proposed for {model_name}.
      #
      # Each entry names a constant tensor that LUT compression can encode and
      # the smallest index_bitwidth that can index each of its value tables.
      # Sizes count the packed indices, the value tables, and the decode
      # header, but not the DECODE operators and decoded tensors that
      # compression adds for each operator that reads a compressed tensor.
      # The whole-model line gives the size of the model compressed as
      # proposed, all of those included. Review the entries and delete
      # those for tensors that should stay uncompressed.
      #
      # Tensor, operator, and buffer numbers refer to the input model.
      # Compression inserts DECODE operators and rewrites tensors and buffers,
      # so numbers in the compressed model differ.""").splitlines()

  if candidates:
    # Tensors sharing a buffer store its data once. Aliases of one shape
    # and layout compress to identical data, which is also stored once.
    original = {c.aliases: c.original_bytes for c in candidates}
    estimated = {
      (c.aliases, tuple(c.shape), c.encoding.axis): c.estimated_bytes
      for c in candidates
    }
    total_original = sum(original.values())
    total_estimated = sum(estimated.values())
    lines.append("#")
    lines.append(
      f"# {_count(len(candidates), 'tensor')}, {total_original:,} -> "
      f"{total_estimated:,} bytes "
      f"({_change(total_original, total_estimated)})"
    )
    lines.append(
      f"# whole model, {model_size:,} -> {compressed_size:,} bytes "
      f"({_change(model_size, compressed_size)})"
    )

  lines.append("")
  lines.append("tensors:")

  for c in candidates:
    lines.append("")
    lines.append(
      f"  # {c.name} {c.type_name} {c.shape}, {', '.join(c.consumers)}"
    )
    encoding = c.encoding
    uniques = _count(encoding.max_unique, "unique value")
    if encoding.axis is not None:
      uniques = (
        f"max {uniques} per table, "
        f"{_count(encoding.tables, 'table')} along axis {encoding.axis}"
      )
    lines.append(
      f"  # {_count(c.elements, 'element')}, {uniques}, "
      f"{c.original_bytes:,} -> {c.estimated_bytes:,} bytes "
      f"({_change(c.original_bytes, c.estimated_bytes)})"
    )
    if c.sharers:
      others = ", ".join(f"subgraph {s} tensor {t}" for s, t in c.sharers)
      lines.append(
        f"  # shares a buffer with {others}; keep or delete "
        "all aliases together"
      )
    lines.append(f"  - subgraph: {c.subgraph}")
    lines.append(f"    tensor: {c.tensor}")
    lines.append("    compression:")
    lines.append("      - lut:")
    lines.append(f"          index_bitwidth: {encoding.bitwidth}")
    if encoding.axis is None:
      lines.append("          per_tensor:")
    else:
      lines.append("          per_channel:")
      lines.append(f"            axis: {encoding.axis}")

  if not candidates:
    lines.append("  []")

  if rejects:
    lines.append("")
    lines.append("# Constant tensors not listed:")
    for r in rejects:
      lines.append(
        f"#   subgraph {r.subgraph} tensor {r.tensor} "
        f"{r.name} {r.type_name} {r.shape}: {r.reason}"
      )

  lines.append("")
  return "\n".join(lines)


def _count(n: int, noun: str) -> str:
  """Formats a count with its noun, e.g. "1 table" or "1,024 elements"."""
  return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _change(original: int, estimated: int) -> str:
  """Describes a size change in bytes and percent, e.g. "16 saved, 50%"."""
  saved = original - estimated
  percent = round(100 * saved / original)
  if saved >= 0:
    return f"{saved:,} saved, {percent}%"
  return f"{-saved:,} larger, {-percent}%"


if __name__ == "__main__":
  sys.exit(main(sys.argv))
