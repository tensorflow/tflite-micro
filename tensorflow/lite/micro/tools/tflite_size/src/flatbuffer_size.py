# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Given a input tflite flatbuffer, outputs a html with flatbuffer size graph"""

import ctypes
import json
import os
import sys

from tflite_micro.tensorflow.lite.micro.tools.tflite_size.src import (
  flatbuffer_size_graph,
)
from tflite_micro.tensorflow.lite.micro.tools.tflite_size.src import (
  flatbuffer_size_graph_html_converter,
)


def _load_flatbuffer_size_library():
  try:
    from rules_python.python.runfiles import runfiles  # pylint: disable=g-import-not-at-top

    r = runfiles.Create()
    path = r.Rlocation(
      "tflite_micro/tensorflow/lite/micro/tools/tflite_size/src/libflatbuffer_size.so"
    )
    if path and os.path.exists(path):
      return ctypes.CDLL(path)
  except Exception:
    pass

  adjacent_path = os.path.join(
    os.path.dirname(__file__), "libflatbuffer_size.so"
  )
  if os.path.exists(adjacent_path):
    return ctypes.CDLL(adjacent_path)

  return ctypes.CDLL("libflatbuffer_size.so")


_lib = _load_flatbuffer_size_library()
_lib.TFLM_FlatbufferSizeConvertToJsonString.restype = ctypes.c_char_p
_lib.TFLM_FlatbufferSizeConvertToJsonString.argtypes = [ctypes.c_char_p]
_lib.TFLM_FlatbufferSizeFreeJsonString.restype = None
_lib.TFLM_FlatbufferSizeFreeJsonString.argtypes = [ctypes.c_char_p]


def convert_tflite_to_html(in_flatbuf):
  """Given a input tflite flatbuffer, returns a html and a json with size info"""
  raw_json = _lib.TFLM_FlatbufferSizeConvertToJsonString(in_flatbuf)
  out_json = raw_json.decode("utf-8")
  json_as_dict = json.loads(out_json)

  formatted_json = json.dumps(json_as_dict)

  graph_builder = flatbuffer_size_graph.FlatbufferSizeGraph()
  graph_builder.create_graph(json_as_dict)

  html_converter = flatbuffer_size_graph_html_converter.HtmlConverter()
  html_string = graph_builder.display_graph(html_converter)

  return html_string, formatted_json


def main(argv):
  try:
    tflite_input = argv[1]
    html_output = argv[2]
  except IndexError:
    print("usage: ", argv[0], " <input tflite> <output html>\n")
  else:
    with open(tflite_input, "rb") as f:
      html_string, _ = convert_tflite_to_html(f.read())
    with open(html_output, "w") as f:
      f.write(html_string)


if __name__ == "__main__":
  main(sys.argv)
