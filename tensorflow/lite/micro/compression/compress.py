# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Model compression library and CLI.

See USAGE.
"""

import os
import sys
import tempfile
from typing import ByteString, Iterable, Type

import absl.app
import absl.flags

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import decode_insert
from tflite_micro.tensorflow.lite.micro.compression import huffman
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import pruning
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.tools import tflite_flatbuffer_align_wrapper

USAGE = f"""\
Usage: compress.py --input <in.tflite> --spec <spec.yaml> [--output <out.tflite>]

Produce a compressed model from the input model by compressing tensors
according to the instructions in the spec file. The spec file lists the tensors
to compress, the compression methods to use on each tensor, and any parameters
for each compression method.

The spec file is a YAML-format file with a dictionary at the root, containing a
key "tensors" with a list of tensors to compress as its value. E.g.:

---
{spec.EXAMPLE_YAML_SPEC}
---

Supported compression methods:

  lut: Look-Up-Table compression. Requires the tensor to have a small number of
       unique values, fewer than or equal to 2**index_bitwidth. LUT compression
       collects these values into a lookup table, and rewrites the tensor as
       bitwidth-wide integer indices into that lookup table.

  huffman: Huffman compression using Xtensa-format decode tables. (Not yet
           implemented.)

  pruning: Pruning (sparsity) compression for sparse tensors. (Not yet
           implemented.)

Compressed models use DECODE operators to decompress tensors at runtime.
"""

# Plugin dispatch table: maps CompressionMethod subclasses to compressor instances
_COMPRESSORS: dict[Type[spec.CompressionMethod], compressor.Compressor] = {
    spec.LookUpTableCompression: lut.LutCompressor(),
    spec.HuffmanCompression: huffman.HuffmanCompressor(),
    spec.PruningCompression: pruning.PruningCompressor(),
}


def _get_compressor(method: spec.CompressionMethod) -> compressor.Compressor:
  """Get the compressor plugin for a given compression method."""
  compressor_instance = _COMPRESSORS.get(type(method))
  if compressor_instance is None:
    raise compressor.CompressionError(
        f"No compressor registered for {type(method).__name__}")
  return compressor_instance


def _apply_flatbuffer_alignment(model_bytes: bytearray) -> bytearray:
  """Applies proper FlatBuffer alignment to a model.

  The Python flatbuffers library doesn't respect `force_align` schema attributes,
  so we use the C++ wrapper which properly handles alignment requirements.

  Args:
    model_bytes: The model flatbuffer to align

  Returns:
    The properly aligned model flatbuffer
  """
  # C++ wrapper requires file paths, not byte buffers
  with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as temp_in:
    temp_in.write(model_bytes)
    temp_in_path = temp_in.name

  with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as temp_out:
    temp_out_path = temp_out.name

  try:
    # Unpack and repack with proper alignment
    tflite_flatbuffer_align_wrapper.align_tflite_model(temp_in_path,
                                                       temp_out_path)

    with open(temp_out_path, 'rb') as f:
      aligned_model = bytearray(f.read())

    return aligned_model
  finally:
    # Clean up temporary files
    if os.path.exists(temp_in_path):
      os.unlink(temp_in_path)
    if os.path.exists(temp_out_path):
      os.unlink(temp_out_path)


def compress(model_in: ByteString, specs: Iterable[spec.Tensor]) -> bytearray:
  """Compresses a model .tflite flatbuffer.

  Compresses tensors according to the given specs and inserts DECODE operators
  to decompress them at runtime.

  Args:
    model_in: the original, uncompressed .tflite flatbuffer
    specs: an iterable of compression specs, see module spec.py

  Returns:
    A compressed flatbuffer with DECODE operators inserted.
  """
  model = model_editor.read(model_in)
  compression_results: dict[tuple[int, int], compressor.CompressionResult] = {}

  for tensor_spec in specs:
    try:
      tensor = model.subgraphs[tensor_spec.subgraph].tensors[
          tensor_spec.tensor]

      # Currently only one compression method per tensor
      if len(tensor_spec.compression) != 1:
        raise compressor.CompressionError(
            "Each tensor must have exactly one compression method")

      method = tensor_spec.compression[0]
      plugin = _get_compressor(method)
      result = plugin.compress(tensor, method)

      # Replace tensor data with encoded data
      tensor.buffer.data = result.encoded_data

      # Store result for DECODE insertion
      compression_results[(tensor_spec.subgraph, tensor_spec.tensor)] = result

    except compressor.CompressionError:
      raise
    except Exception as e:
      raise compressor.CompressionError(
          f"error compressing {tensor_spec}") from e

  # Insert DECODE operators into the graph
  decode_insert.insert_decode_operators(model, compression_results)

  # Build the model and apply proper alignment
  unaligned_model = model.build()
  return _apply_flatbuffer_alignment(unaligned_model)


def _fail_w_usage() -> int:
  absl.app.usage()
  return 1


FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("input", None, help="uncompressed .tflite flatbuffer")
absl.flags.DEFINE_string("spec", None, help="specfile (see module spec.py)")
absl.flags.DEFINE_string("output", None, help="compressed .tflite flatbuffer")


def main(argv):
  if len(argv) > 1:
    # no positional arguments accepted
    return _fail_w_usage()

  in_path = FLAGS.input
  if in_path is None:
    return _fail_w_usage()
  else:
    with open(in_path, "rb") as in_file:
      in_model = in_file.read()

  spec_path = FLAGS.spec
  if spec_path is None:
    return _fail_w_usage()
  else:
    with open(spec_path, "r") as spec_file:
      specs = spec.parse_yaml(spec_file.read())

  out_path = FLAGS.output
  if out_path is None:
    out_path = in_path.split(".tflite")[0] + ".compressed.tflite"

  compressed = compress(in_model, specs)

  with open(out_path, "wb") as out_file:
    out_file.write(compressed)

  return 0


if __name__ == "__main__":
  sys.modules['__main__'].__doc__ = USAGE  # for absl's use
  absl.app.run(main)
