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

import bitarray
import bitarray.util
import pprint
import textwrap
import os
import sys

import lib
from tensorflow.lite.micro.compression import metadata_py_generated as compression_schema
from tensorflow.lite.python import schema_py_generated as tflite_schema

import absl.app


def unpack_list(source):
  result = []
  for index, s in enumerate(source):
    d = {"_index": index} | vars(s)
    result.append(d)
  return result


def unpack_operators(operators):
  result = []
  for index, o in enumerate(operators):
    d = {
        "_index": index,
        "opcode_index": o.opcodeIndex,
        "inputs": unpack_array(o.inputs),
        "outputs": unpack_array(o.outputs),
    }
    result.append(d)
  return result


def unpack_TensorType(type):
  attrs = [
      attr for attr in dir(tflite_schema.TensorType)
      if not attr.startswith("__")
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


def unpack_tensors(tensors):
  result = []
  for index, t in enumerate(tensors):
    d = {
        "_tensor": index,
        "name": _decode_name(t.name),
        "type": unpack_TensorType(t.type),
        "shape": unpack_array(t.shape),
        "buffer": t.buffer,
    }

    if t.isVariable:
      d["is_variable"] = True
    else:
      # don't display this unusual field
      pass

    if t.quantization is not None and t.quantization.scale is not None:
      d["quantization"] = {
          "scale": unpack_array(t.quantization.scale),
          "zero": unpack_array(t.quantization.zeroPoint),
          "dimension": t.quantization.quantizedDimension,
      }
    result.append(d)
  return result


def unpack_subgraphs(subgraphs):
  result = []
  for index, s in enumerate(subgraphs):
    d = {
        "_index": index,
        "name": _decode_name(s.name),
        # "inputs": s.inputs,
        # "outputs": s.outputs,
        "operators": unpack_operators(s.operators),
        "tensors": unpack_tensors(s.tensors),
    }
    result.append(d)
  return result


def unpack_metadata(metadata):
  if metadata is None:
    return None
  return [{"name": _decode_name(m.name), "buffer": m.buffer} for m in metadata]


def unpack_lut_tensors(lut_tensors):
  result = []
  for index, t in enumerate(lut_tensors):
    result.append({
        "tensor": t.tensor,
        "value_buffer": t.valueBuffer,
        "index_bitwidth": t.indexBitwidth
    })
  return result


def unpack_compression_metadata(buffer):
  buffer = bytes(buffer.data)
  metadata = compression_schema.MetadataT.InitFromPackedBuf(buffer, 0)
  result = []
  for index, s in enumerate(metadata.subgraphs):
    d = {"_index": index, "lut_tensors": unpack_lut_tensors(s.lutTensors)}
    result.append(d)
  return {"subgraphs": result}


def unpack_array(a):
  try:
    # Avoid printing as numpy arrays if possible. The pprint module does not
    # format them well.
    a = a.tolist()
  except AttributeError:
    pass
  return a


def is_compressed_buffer(buffer_index, unpacked_metadata):
  if unpacked_metadata is None:
    return False, None, None
  for subgraph in unpacked_metadata["subgraphs"]:
    lut_list = subgraph["lut_tensors"]
    subgraph_index = subgraph["_index"]
    item = next(
        (item for item in lut_list if item["value_buffer"] == buffer_index),
        None)
    if item is not None:
      return True, item, subgraph_index
  return False, None, None


def unpack_indices(buffer, lut_data):
  bstring = bitarray.bitarray()
  bstring.frombytes(bytes(buffer.data))
  bitwidth = lut_data["index_bitwidth"]
  indices = []
  while len(bstring) > 0:
    indices.append(bitarray.util.ba2int(bstring[0:bitwidth]))
    del bstring[0:bitwidth]
  return indices


def unpack_buffers(model, compression_metadata=None, unpacked_metadata=None):
  buffers = model.buffers
  result = []
  for index, b in enumerate(buffers):
    d = {"_buffer": index}
    d = d | {"_bytes": len(b.data) if b.data is not None else 0}
    d = d | {"data": unpack_array(b.data)}
    if index == compression_metadata:
      if unpacked_metadata is not None:
        unpacked = unpacked_metadata
      else:
        unpacked = {"parse error"}
      d = d | {"_compression_metadata_decoded": unpacked}
    else:
      is_compressed, lut_data, subgraph_index = is_compressed_buffer(
          index, unpacked_metadata)
      if is_compressed:
        tensor_index = lut_data["tensor"]
        tensor_buffer_index = model.subgraphs[subgraph_index].tensors[
            tensor_index].buffer
        tensor_buffer = model.buffers[tensor_buffer_index]
        d = d | {"indices": unpack_indices(tensor_buffer, lut_data)}
    result.append(d)
  return result


def get_compression_metadata_buffer(model):
  # Return the metadata buffer data or None
  if model.metadata is None:
    return None
  for item in model.metadata:
    if _decode_name(item.name) == "COMPRESSION_METADATA":
      return item.buffer
  return None


def create_dictionary(flatbuffer: memoryview) -> dict:
  """Returns a human-readable dictionary from the provided model flatbuffer.

  This function transforms a .tflite model flatbuffer into a Python dictionary.
  When pretty-printed, this dictionary offers an easily interpretable view of
  the model.
  """
  model = tflite_schema.ModelT.InitFromPackedBuf(flatbuffer, 0)

  comp_metadata_index = get_compression_metadata_buffer(model)
  comp_metadata_unpacked = None
  if comp_metadata_index is not None:
    try:
      comp_metadata_unpacked = unpack_compression_metadata(
          model.buffers[comp_metadata_index])
    except TypeError:
      pass

  output = {
      "description": model.description,
      "version": model.version,
      "operator_codes": unpack_list(model.operatorCodes),
      "metadata": unpack_metadata(model.metadata),
      "subgraphs": unpack_subgraphs(model.subgraphs),
      "buffers":
      unpack_buffers(model, comp_metadata_index, comp_metadata_unpacked),
  }

  return output


def main(argv):
  path = argv[1]
  with open(path, 'rb') as flatbuffer:
    d = create_dictionary(memoryview(flatbuffer.read()))
    pprint.pprint(d, width=90, sort_dicts=False, compact=True)


if __name__ == "__main__":
  name = os.path.basename(sys.argv[0])
  usage = textwrap.dedent(f"""\
      Usage: {name} <MODEL>
      Print a visualization of a .tflite model.""")
  sys.modules['__main__'].__doc__ = usage
  absl.app.run(main)
