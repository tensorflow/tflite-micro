# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

# This development tool prints compressed and uncompressed .tflite models to
# stdout in a human-readable, searchable, structured, text format. Helpful
# annotations (indexes of lists, names of operators, etc.) derived from the
# model are added as virtual fields with names beginning with an _underscore.
#
# Example usage:
#   bazel run //tensorflow/lite/micro/compression:view -- $(realpath model.tflite)
#
# Theory of operation:
# Convert the model into a Python dictionary, expressing the hierarchical nature
# of the model, and pretty print the dictionary. Please extend as needed for
# your use case.

import numpy as np
import os
import pprint
import sys
import textwrap

from tflite_micro.tensorflow.lite.micro.python import (
  schema_py_generated as tflite_schema,
)

# Detect if running under Bazel by checking for BAZEL environment variables
is_bazel = 'BUILD_WORKING_DIRECTORY' in os.environ or 'BAZEL_TEST' in os.environ

if is_bazel:
  USAGE = textwrap.dedent("""\
    Usage: bazel run //tensorflow/lite/micro/compression:view -- <MODEL_PATH>
    
    Print a human-readable visualization of a .tflite model.
    Note: When running through Bazel, MODEL_PATH must be an absolute path.
    
    Example: bazel run //tensorflow/lite/micro/compression:view -- $(realpath model.tflite)""")
else:
  USAGE = textwrap.dedent(f"""\
    Usage: {os.path.basename(sys.argv[0])} <MODEL_PATH>
    
    Print a human-readable visualization of a .tflite model.""")


def print_model(model_path):
  with open(model_path, 'rb') as flatbuffer:
    d = create_dictionary(memoryview(flatbuffer.read()))
    with np.printoptions(linewidth=78, suppress=True):
      pprint.pprint(d, width=78, sort_dicts=False)


def main(argv):
  try:
    model_path = argv[1]
  except IndexError:
    sys.stderr.write(USAGE)
    sys.exit(1)

  print_model(model_path)


def unpack_operators(
  model: tflite_schema.ModelT, operators: list[tflite_schema.OperatorT]
):
  result = []
  for index, op in enumerate(operators):
    opcode = model.operatorCodes[op.opcodeIndex]
    name = OPERATOR_NAMES[opcode.builtinCode]
    d = {
      "_operator": index,
      "opcode_index": op.opcodeIndex,
      "_opcode_name": name,
      "inputs": op.inputs,
      "outputs": op.outputs,
    }
    result.append(d)
  return result


def unpack_TensorType(type):
  attrs = [
    attr for attr in dir(tflite_schema.TensorType) if not attr.startswith("__")
  ]
  lut = {getattr(tflite_schema.TensorType, attr): attr for attr in attrs}
  return lut[type]


def _decode_name(name):
  """Returns name as a str or 'None'.

  The flatbuffer library returns names as bytes objects or None. This function
  returns a str, decoded from the bytes object, or None.
  """
  if name is None:
    return None
  else:
    return str(name, encoding="utf-8")


OPERATOR_NAMES = {
  code: name for name, code in tflite_schema.BuiltinOperator.__dict__.items()
}


def unpack_tensors(tensors):
  result = []
  for index, t in enumerate(tensors):
    d = {
      "_tensor": index,
      "name": _decode_name(t.name),
      "type": unpack_TensorType(t.type),
      "shape": t.shape,
      "buffer": t.buffer,
    }

    if t.isVariable:
      d["is_variable"] = True
    else:
      # don't display this unusual field
      pass

    if t.quantization is not None and t.quantization.scale is not None:
      d["quantization"] = {
        "scale": t.quantization.scale,
        "zero": t.quantization.zeroPoint,
        "dimension": t.quantization.quantizedDimension,
      }
    result.append(d)

  return result


def unpack_subgraphs(model: tflite_schema.ModelT):
  result = []
  for index, s in enumerate(model.subgraphs):
    d = {
      "_subgraph": index,
      "_operator_count": len(s.operators),
      "_tensor_count": len(s.tensors),
      "name": _decode_name(s.name),
      "operators": unpack_operators(model, s.operators),
      "tensors": unpack_tensors(s.tensors),
    }
    result.append(d)
  return result


def unpack_opcodes(opcodes: list[tflite_schema.OperatorCodeT]) -> list:
  result = []
  for index, opcode in enumerate(opcodes):
    d: dict = {
      "_opcode_index": index,
      "_name": OPERATOR_NAMES[opcode.builtinCode],
      "builtin_code": opcode.builtinCode,
      "version": opcode.version,
    }
    if opcode.customCode is not None:
      d["custom_code"] = opcode.customCode
      del d["_name"]
    result.append(d)
  return result


def unpack_metadata(model: tflite_schema.ModelT):
  entries = []

  if model.metadata is None:
    return entries

  for m in model.metadata:
    d = {"name": _decode_name(m.name), "buffer": m.buffer}
    entries.append(d)

  return entries


def unpack_buffers(model):
  buffers = []
  for index, buffer in enumerate(model.buffers):
    native = {
      "_buffer": index,
      "_bytes": len(buffer.data) if buffer.data is not None else 0,
      "data": buffer.data,
    }
    buffers.append(native)

  return buffers


def create_dictionary(flatbuffer: memoryview) -> dict:
  """Returns a human-readable dictionary from the provided model flatbuffer.

  This function transforms a .tflite model flatbuffer into a Python dictionary.
  When pretty-printed, this dictionary offers an easily interpretable view of
  the model.
  """
  model = tflite_schema.ModelT.InitFromPackedBuf(flatbuffer, 0)

  output = {
    "description": model.description,
    "version": model.version,
    "operator_codes": unpack_opcodes(model.operatorCodes),
    "metadata": unpack_metadata(model),
    "subgraphs": unpack_subgraphs(model),
    "buffers": unpack_buffers(model),
  }

  return output


if __name__ == "__main__":
  main(sys.argv)
