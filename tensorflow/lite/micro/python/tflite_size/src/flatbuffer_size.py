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
import json

from tflite_micro.tensorflow.lite.micro.python.tflite_size.src import flatbuffer_size_wrapper_pybind
from tflite_micro.tensorflow.lite.micro.python.tflite_size.src import flatbuffer_size_graph
from tflite_micro.tensorflow.lite.micro.python.tflite_size.src import flatbuffer_size_graph_html_converter


def convert_tflite_to_html(in_filename, out_html_file, out_json_file=''):
  with open(in_filename, 'rb') as f:
    in_flatbuf = f.read()

    sizeWrapper = flatbuffer_size_wrapper_pybind.FlatbufferSize()
    outJson = sizeWrapper.convertToJsonString(in_flatbuf)
    jsonAsDict = json.loads(outJson)

    # Write json output to compare with golden vector file
    if out_json_file:
      formatedJson = json.dumps(jsonAsDict)
      with open(out_json_file, 'w') as f:
        f.write(formatedJson)

    graphBuilder = flatbuffer_size_graph.FlatbufferSizeGraph()
    graphBuilder.createGraph(jsonAsDict)

    htmlConverter = flatbuffer_size_graph_html_converter.HtmlConverter()
    htmlString = graphBuilder.displayGraph(htmlConverter)

    # Write html output to compare with golden vector file
    with open(out_html_file, 'w') as f:
      f.write(htmlString)
