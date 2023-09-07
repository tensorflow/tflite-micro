# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
""" Generates C/C++ source code capable of performing inference for a model. """

import os

from absl import app
from absl import flags
from collections.abc import Sequence

from tflite_micro.codegen import inference_generator
from tflite_micro.codegen import graph
from tflite_micro.codegen.preprocessor import preprocessor_schema_py_generated as preprocessor_fb
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils

# Usage information:
# Default:
#   `bazel run codegen:code_generator -- \
#        --model=</path/to/my_model.tflite> \
#        --preprocessed_data=</path/to/preprocesser_output>`
# Output will be located at: /path/to/my_model.h|cc

_MODEL_PATH = flags.DEFINE_string(name="model",
                                  default=None,
                                  help="Path to the TFLite model file.",
                                  required=True)

_PREPROCESSED_DATA_PATH = flags.DEFINE_string(
    name="preprocessed_data",
    default=None,
    help="Path to output of codegen_preprocessor.",
    required=True)

_OUTPUT_DIR = flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="Path to write generated source to. Leave blank to use 'model' path.",
    required=False)

_OUTPUT_NAME = flags.DEFINE_string(
    name="output_name",
    default=None,
    help=("The output basename for the generated .h/.cc. Leave blank to use "
          "'model' basename."),
    required=False)


def _read_preprocessed_data(
    preprocessed_data_file: str) -> preprocessor_fb.DataT:
  with open(preprocessed_data_file, 'rb') as file:
    data_byte_array = bytearray(file.read())
  return preprocessor_fb.DataT.InitFromObj(
      preprocessor_fb.Data.GetRootAs(data_byte_array, 0))


def main(argv: Sequence[str]) -> None:
  output_dir = _OUTPUT_DIR.value or os.path.dirname(_MODEL_PATH.value)
  output_name = _OUTPUT_NAME.value or os.path.splitext(
      os.path.basename(_MODEL_PATH.value))[0]

  model = flatbuffer_utils.read_model(_MODEL_PATH.value)
  preprocessed_data = _read_preprocessed_data(_PREPROCESSED_DATA_PATH.value)

  print("Generating inference code for model:\n"
        "  model: {}\n"
        "  preprocessed_model: {}\n".format(
            _MODEL_PATH.value,
            preprocessed_data.inputModelPath.decode('utf-8')))

  inference_generator.generate(output_dir, output_name,
                               graph.OpCodeTable([model]), graph.Graph(model))


if __name__ == "__main__":
  app.run(main)
