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
"""Build and compress a model with the model editor.

This example builds a small model from scratch with model_editor: a
FULLY_CONNECTED layer feeding a CONV_2D whose filter holds float16
values clustered to 8 uniques. It writes that model as a .tflite file,
then compresses the filter with 3-bit LUT compression, which rewrites
the filter as packed indices and inserts a DECODE operator between it
and the CONV_2D. It writes the compressed model as a second .tflite
file and prints both operator graphs.

The example then builds the compressed model a second time, assembling
the encoded filter, the ancillary data, and the DECODE operator
directly with the editor. It checks that this hand-built model matches
the tool-built model compress() produced, and writes it as a third
.tflite file.

The builders repeat common code instead of factoring it out, so each
one reads on its own from top to bottom, like a unit test.

Run under bazel, which writes the .tflite files to the invocation
directory by default:

  bazel run //tensorflow/lite/micro/compression/tutorial:model_editor_example

An optional argument overrides the output directory:

  bazel run //tensorflow/lite/micro/compression/tutorial:model_editor_example \\
      -- /path/to/output
"""

import os
import sys
import tempfile

import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import decode_insert
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import spec_builder
from tflite_micro.tensorflow.lite.micro.tools import (
  tflite_flatbuffer_align_wrapper,
)
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


def build_model() -> bytearray:
  """Build the uncompressed model and return its flatbuffer.

  The model is a fully connected layer feeding a convolution:

    fc_input (1, 8)
      | FULLY_CONNECTED, fc_weights (64, 8)
    fc_output (1, 4, 4, 4)
      | CONV_2D, conv_filter (16, 3, 3, 4) float16, no bias
    conv_output (1, 4, 4, 16)
  """
  fc_input = model_editor.Tensor(
    shape=(1, 8),
    dtype=tflite.TensorType.FLOAT32,
    name="fc_input",
  )
  fc_weights = model_editor.Tensor(
    shape=(64, 8),
    dtype=tflite.TensorType.FLOAT32,
    data=np.linspace(-1.0, 1.0, num=64 * 8, dtype=np.float32).reshape(64, 8),
    name="fc_weights",
  )

  # The FC output doubles as the CONV_2D input, so declare it with the
  # conv's 4D layout. A converter would insert a RESHAPE here; this
  # example elides it for brevity.
  fc_output = model_editor.Tensor(
    shape=(1, 4, 4, 4),
    dtype=tflite.TensorType.FLOAT32,
    name="fc_output",
  )

  fc_op = model_editor.Operator(
    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
    inputs=[fc_input, fc_weights],
    outputs=[fc_output],
    options=tflite.FullyConnectedOptionsT(),
  )

  # Eight distinct float16 values stand in for the codebook of a filter
  # clustered during training. Cycling through the codebook fills the
  # filter with exactly 8 unique values, so 3-bit LUT indices can
  # encode it. The filter is large enough that the savings outweigh the
  # DECODE operator and ancillary data the compressed model adds.
  codebook = np.array(
    [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0], dtype=np.float16
  )
  conv_filter = model_editor.Tensor(
    shape=(16, 3, 3, 4),
    dtype=tflite.TensorType.FLOAT16,
    data=np.resize(codebook, (16, 3, 3, 4)),
    name="conv_filter",
  )
  conv_output = model_editor.Tensor(
    shape=(1, 4, 4, 16),
    dtype=tflite.TensorType.FLOAT32,
    name="conv_output",
  )
  conv_op = model_editor.Operator(
    opcode=tflite.BuiltinOperator.CONV_2D,
    # None marks the absent optional bias input.
    inputs=[fc_output, conv_filter, None],
    outputs=[conv_output],
    options=tflite.Conv2DOptionsT(
      padding=tflite.Padding.SAME, strideW=1, strideH=1
    ),
  )

  model = model_editor.Model(
    subgraphs=[
      model_editor.Subgraph(
        tensors=[fc_weights, conv_filter],
        inputs=[fc_input],
        outputs=[conv_output],
        operators=[fc_op, conv_op],
      )
    ]
  )
  return model.build()


def build_compressed_model() -> bytearray:
  """Build the compressed model and return its flatbuffer.

  The model is a fully connected layer feeding a convolution. The
  convolution reads its filter from a DECODE operator, which decodes
  it from LUT-compressed constants:

    fc_input (1, 8)
      | FULLY_CONNECTED, fc_weights (64, 8)
    fc_output (1, 4, 4, 4)
      | CONV_2D, filter from DECODE below, no bias
    conv_output (1, 4, 4, 16)

    conv_filter (216,) uint8, conv_filter_ancillary (32,) uint8
      | TFLM_DECODE
    conv_filter_decoded (16, 3, 3, 4) float16

  A compressed graph needs nothing special from the editor. The
  encoded filter and its ancillary data are constant tensors, and
  DECODE is a custom operator connecting them to the decoded tensor
  the convolution reads. Use the LUT compressor plugin for the
  encoded bytes and assemble the graph by hand.
  """
  # The FC layer is the same as in build_model.
  fc_input = model_editor.Tensor(
    shape=(1, 8),
    dtype=tflite.TensorType.FLOAT32,
    name="fc_input",
  )
  fc_weights = model_editor.Tensor(
    shape=(64, 8),
    dtype=tflite.TensorType.FLOAT32,
    data=np.linspace(-1.0, 1.0, num=64 * 8, dtype=np.float32).reshape(64, 8),
    name="fc_weights",
  )
  fc_output = model_editor.Tensor(
    shape=(1, 4, 4, 4),
    dtype=tflite.TensorType.FLOAT32,
    name="fc_output",
  )
  fc_op = model_editor.Operator(
    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
    inputs=[fc_input, fc_weights],
    outputs=[fc_output],
    options=tflite.FullyConnectedOptionsT(),
  )

  # Encode the filter values outside any model. The values are the
  # same as in build_model, so the two compressed models can be
  # compared. The compressor plugin takes a tensor and returns the
  # packed indices and the ancillary data, a DCM header followed by
  # the value table.
  codebook = np.array(
    [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0], dtype=np.float16
  )
  values = model_editor.Tensor(
    shape=(16, 3, 3, 4),
    dtype=tflite.TensorType.FLOAT16,
    data=np.resize(codebook, (16, 3, 3, 4)),
    name="conv_filter",
  )
  compression = lut.LutCompressor().compress(
    values, spec.LookUpTableCompression(index_bitwidth=3, mode=spec.PerTensor())
  )
  encoded_data = bytes(compression.encoded_data)
  ancillary_data = bytes(compression.ancillary_data)

  # The encoded filter and the ancillary data are plain uint8
  # constants.
  conv_filter = model_editor.Tensor(
    shape=(len(encoded_data),),
    dtype=tflite.TensorType.UINT8,
    data=encoded_data,
    name="conv_filter",
  )
  ancillary = model_editor.Tensor(
    shape=(len(ancillary_data),),
    dtype=tflite.TensorType.UINT8,
    data=ancillary_data,
    name="conv_filter_ancillary",
  )

  # The DECODE output has the shape and type of the original filter
  # and no data; the operator produces the values at runtime.
  decoded = model_editor.Tensor(
    shape=(16, 3, 3, 4),
    dtype=tflite.TensorType.FLOAT16,
    name="conv_filter_decoded",
  )
  decode_op = model_editor.Operator(
    opcode=tflite.BuiltinOperator.CUSTOM,
    custom_code=decode_insert.DECODE_CUSTOM_OP_NAME,
    inputs=[conv_filter, ancillary],
    outputs=[decoded],
  )

  # The convolution reads the decoded tensor where the uncompressed
  # model reads the filter itself.
  conv_output = model_editor.Tensor(
    shape=(1, 4, 4, 16),
    dtype=tflite.TensorType.FLOAT32,
    name="conv_output",
  )
  conv_op = model_editor.Operator(
    opcode=tflite.BuiltinOperator.CONV_2D,
    # None marks the absent optional bias input.
    inputs=[fc_output, decoded, None],
    outputs=[conv_output],
    options=tflite.Conv2DOptionsT(
      padding=tflite.Padding.SAME, strideW=1, strideH=1
    ),
  )

  model = model_editor.Model(
    subgraphs=[
      model_editor.Subgraph(
        tensors=[fc_weights, conv_filter, ancillary],
        inputs=[fc_input],
        outputs=[conv_output],
        operators=[fc_op, decode_op, conv_op],
      )
    ]
  )
  return model.build()


def compress_filter(flatbuffer: bytes) -> bytearray:
  """Compress the conv filter and return the compressed flatbuffer.

  compress() takes a built model and a compression spec, rewrites the
  listed tensors in place, and inserts the DECODE operators itself,
  producing the same graph build_compressed_model assembles by hand.
  """
  # The compression spec names tensors by index. Read the built model
  # back and look the filter up by name rather than hard-coding its
  # position.
  model = model_editor.read(flatbuffer)
  filter_index = model.subgraphs[0].tensor_by_name("conv_filter").index

  # The filter is unquantized, so one value table covers it.
  specs = (
    spec_builder.SpecBuilder()
    .add_tensor(subgraph=0, tensor=filter_index)
    .with_lut(index_bitwidth=3, mode=spec.PerTensor())
    .build()
  )
  return compress.compress(flatbuffer, specs)


_OPERATOR_NAMES = {
  value: name
  for name, value in vars(tflite.BuiltinOperator).items()
  if not name.startswith("_")
}


def describe(flatbuffer: bytes, title: str) -> None:
  """Print a model's operators in graph order."""
  model = model_editor.read(flatbuffer)
  print(f"{title} ({len(flatbuffer)} bytes):")
  for op in model.subgraphs[0].operators:
    if op.opcode == tflite.BuiltinOperator.CUSTOM:
      name = op.custom_code
    else:
      name = _OPERATOR_NAMES[op.opcode]
    inputs = ", ".join("<none>" if t is None else t.name for t in op.inputs)
    outputs = ", ".join(t.name for t in op.outputs)
    print(f"  {name}({inputs}) -> {outputs}")


def graph_summary(flatbuffer: bytes):
  """Summarize a model's graph, invariant to tensor numbering.

  Two constructions of the same graph can order the tensor list
  differently, so compare operators by the names of the tensors they
  connect, and tensors by name, type, shape, and contents.
  """
  model = model_editor.read(flatbuffer)
  subgraph = model.subgraphs[0]
  operators = []
  for op in subgraph.operators:
    if op.opcode == tflite.BuiltinOperator.CUSTOM:
      name = op.custom_code
    else:
      name = _OPERATOR_NAMES[op.opcode]
    operators.append(
      (
        name,
        tuple(None if t is None else t.name for t in op.inputs),
        tuple(t.name for t in op.outputs),
      )
    )
  tensors = {
    t.name: (t.dtype, t.shape, bytes(t.buffer) if t.buffer else None)
    for t in subgraph.tensors
  }
  return operators, tensors


def write_tflite(flatbuffer: bytes, path: str) -> None:
  """Write a model flatbuffer to path as a finished .tflite file.

  The Python flatbuffers library does not respect the schema's
  force_align attributes, so repack through the C++ wrapper, which
  aligns tensor buffers properly. The wrapper works on files, hence
  the temporary.
  """
  with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as temp:
    temp.write(flatbuffer)
    temp_path = temp.name
  try:
    tflite_flatbuffer_align_wrapper.align_tflite_model(temp_path, path)
  finally:
    os.unlink(temp_path)


def main() -> int:
  # Under bazel run, the working directory is inside the runfiles tree;
  # default to the directory bazel was invoked from.
  if len(sys.argv) > 1:
    out_dir = sys.argv[1]
  else:
    out_dir = os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd())

  flatbuffer = bytes(build_model())
  tool_built = bytes(compress_filter(flatbuffer))
  hand_built = bytes(build_compressed_model())

  describe(flatbuffer, "uncompressed")
  describe(tool_built, "tool-built")

  if graph_summary(hand_built) == graph_summary(tool_built):
    print("hand-built model matches the tool-built model")
  else:
    describe(hand_built, "hand-built (differs from the tool-built model)")

  for name, data in (
    ("model_editor_example.tflite", flatbuffer),
    ("model_editor_example.tool_built.tflite", tool_built),
    ("model_editor_example.hand_built.tflite", hand_built),
  ):
    path = os.path.join(out_dir, name)
    write_tflite(data, path)
    print(f"wrote {path}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
