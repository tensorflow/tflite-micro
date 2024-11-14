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
"""
Usage:
  bazel run tensorflow/lite/micro/tools:discretize -- \\
      $(realpath <input.tflite>) [<output.tflite>]

Given a model, rewrite the elements of certain tensors to a small number of
discrete values. This is the first stage of model compression using
look-up-table tensors. A future stage, in a different program, transforms the
rewritten tensors into look-up-table tensors, wherein the elements are reduced
to indices into a value table containing the discrete values.

This program is meant as a test and reference implementation for other
first-stage programs which similarly rewrite elements, using more sophesticated
methods for determining the discrete values.
"""

from tensorflow.lite.python import schema_py_generated as tflite_schema

import absl.app
import flatbuffers
import numpy as np
import sklearn.cluster
import struct
import sys

TENSOR_TYPE_TO_STRUCT_FORMAT = {
    tflite_schema.TensorType.INT8: "b",
    tflite_schema.TensorType.INT16: "h",
    tflite_schema.TensorType.INT32: "i",
    tflite_schema.TensorType.FLOAT32: "f",
}


def unpack_buffer_data(data, struct_format):
  little_endian = "<"
  unpacker = struct.Struct(little_endian + struct_format)
  values = [v[0] for v in unpacker.iter_unpack(bytes(data))]
  return values


def bin_and_quant(sequence, num_values):
  """Quantize a sequence of integers, minimizing the total error using k-means
  clustering.

  Parameters:
    sequence :list - a sequence of integers to be quanized
    num_values :int - the number of quantization levels

  Returns:
    The input sequence, with all values quantized to one of the discovered
    quantization levels.
  """
  sequence = np.array(sequence).reshape(-1, 1)
  kmeans = sklearn.cluster.KMeans(n_clusters=num_values,
                                  random_state=0).fit(sequence)
  indices = kmeans.predict(sequence).tolist()
  values = kmeans.cluster_centers_.flatten()
  values = np.round(values).astype(int).tolist()
  quantized = [values[i] for i in indices]
  return quantized


def replace_buffer_data(buffer, values, format):
  new = bytearray()
  little_endian = "<"
  packer = struct.Struct(little_endian + format)
  for v in values:
    new.extend(packer.pack(v))

  assert (len(buffer.data) == len(new))
  buffer.data = new


def discretize_tensor(tensor, buffer):
  format = TENSOR_TYPE_TO_STRUCT_FORMAT[tensor.type]
  values = unpack_buffer_data(buffer.data, format)
  levels = 4
  if len(values) > levels:
    discretized = bin_and_quant(values, 4)
    replace_buffer_data(buffer, discretized, format)


def map_actionable_opcodes(model):
  """Sparsely map operator code indices to indices of input tensors to
  discretize."""

  actionable_operators = {
      tflite_schema.BuiltinOperator.FULLY_CONNECTED: (1, 2)
  }

  opcodes = {}
  for index, operator_code in enumerate(model.operatorCodes):
    inputs = actionable_operators.get(operator_code.builtinCode, None)
    if inputs is not None:
      opcodes[index] = inputs

  return opcodes


def discretize(model):
  # Discretize the input tensors of which operator_codes?
  actionable_opcodes = map_actionable_opcodes(model)

  # Walk graph nodes (operators) and build list of tensors to discretize
  tensors = set()
  for subgraph_id, subgraph in enumerate(model.subgraphs):
    for operator_id, operator in enumerate(subgraph.operators):
      inputs = actionable_opcodes.get(operator.opcodeIndex, None)
      if inputs is not None:
        for input in (operator.inputs[i] for i in inputs):
          tensors.add(subgraph.tensors[input])

  # Discretize tensors
  for t in tensors:
    discretize_tensor(t, model.buffers[t.buffer])

  return model


def read_model(path):
  with open(path, 'rb') as file:
    buffer = bytearray(file.read())
  return tflite_schema.ModelT.InitFromPackedBuf(buffer, 0)


def write_model(model, path):
  builder = flatbuffers.Builder(32)
  root = model.Pack(builder)
  builder.Finish(root)
  buffer: bytearray = builder.Output()

  with open(path, 'wb') as file:
    file.write(buffer)


def main(argv) -> None:
  try:
    input_path = argv[1]
  except IndexError:
    absl.app.usage()
    return 1

  try:
    output_path = argv[2]
  except IndexError:
    output_path = input_path.split(".tflite")[0] + ".discretized.tflite"

  print(f"discretizing {input_path} to {output_path}")

  model = read_model(input_path)
  model = discretize(model)
  write_model(model, output_path)

  return 0


if __name__ == "__main__":
  rc = absl.app.run(main)
  sys.exit(rc)
