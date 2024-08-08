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


import pprint

from tensorflow.lite.micro.compression import metadata_flatbuffer_py_generated as compression_schema
from tensorflow.lite.python import schema_py_generated as tflite_schema


def read_model(path):
  with open(path, 'rb') as file:
    buffer = bytearray(file.read())
  return tflite_schema.ModelT.InitFromPackedBuf(buffer, 0)


def unpack_list(source):
  result = []
  for index, s in enumerate(source):
    d = {"_index": index} | vars(s)
    result.append(d)
  return result


def unpack_operators(operators):
  result = []
  for index, o in enumerate(operators):
    d = {"_index": index,
         "opcode_index": o.opcodeIndex,
         "inputs": unpack_array(o.inputs),
         "outputs": unpack_array(o.outputs),
         }
    result.append(d)
  return result


def unpack_TensorType(type):
  attrs = [attr for attr in dir(tflite_schema.TensorType) if not
           attr.startswith("__")]
  lut = {getattr(tflite_schema.TensorType, attr): attr for attr in attrs}
  return lut[type]


def unpack_tensors(tensors):
  result = []
  for index, t in enumerate(tensors):
    d = {"_index": index,
         "name": t.name.decode("utf-8"),
         "type": unpack_TensorType(t.type),
         "shape": unpack_array(t.shape),
         "quantization": [unpack_array(t.quantization.scale), unpack_array(t.quantization.zeroPoint)],
         "buffer": t.buffer,
         }
    result.append(d)
  return result


def unpack_subgraphs(subgraphs):
  result = []
  for index, s in enumerate(subgraphs):
    d = {"_index": index,
         "name": s.name,
         # "inputs": s.inputs,
         # "outputs": s.outputs,
         "operators": unpack_operators(s.operators),
         "tensors": unpack_tensors(s.tensors),
         }
    result.append(d)
  return result


def unpack_metadata(metadata):
  return [{"name": m.name.decode("utf-8"), "buffer": m.buffer} for m in
          metadata]


def unpack_compression_metadata(buffer):
  metadata = compression_schema.MetadataT.InitFromPackedBuf(buffer, 0)
  result = []
  for index, t in enumerate(metadata.lutTensors):
    d = {"_index": index,
         "subgraph": t.subgraph,
         "tensor": t.tensor,
         "indexBitwidth": t.indexBitwidth,
         "indexBuffer": t.indexBuffer,
         "valueBuffer": t.valueBuffer,
         }
    result.append(d)
  return {"lut_tensors": result}


def unpack_array(a):
  try:
    # Avoid printing as numpy arrays if possible. The pprint module does not
    # format them well.
    a = a.tolist()
  except AttributeError:
    pass
  return a


def unpack_buffers(buffers, compression_metadata=None):
  result = []
  for index, b in enumerate(buffers):
    d = {"_index": index}
    d = d | {"data": unpack_array(b.data)}
    if index == compression_metadata: d = d | {"_compression_metadata_decoded":
                                               unpack_compression_metadata(bytes(b.data))}
    result.append(d)
  return result


def get_compression_metadata_buffer(model):
  # Return the metadata buffer data or None
  for item in model.metadata:
    if item.name.decode("utf-8") == "COMPRESSION_METADATA":
      return item.buffer
  else:
    return None


def print_model(model, format=None):
  output = {
      "description": model.description.decode("utf-8"),
      "version": model.version,
      "operator_codes": unpack_list(model.operatorCodes),
      "metadata": unpack_metadata(model.metadata),
      "subgraphs": unpack_subgraphs(model.subgraphs),
      "buffers": unpack_buffers(model.buffers,
                                get_compression_metadata_buffer(model)),
      }

  pprint.pprint(output, width=90, sort_dicts=False, compact=True)


def main(argv=None):
  filename = argv[1]
  model = read_model(filename)
  print_model(model)


if __name__ == "__main__":
  import sys
  main(sys.argv)
