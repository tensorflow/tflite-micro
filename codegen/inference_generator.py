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
""" Generates C/C++ inference source code. """

import pathlib

from mako import template
from typing import TypedDict

from tflite_micro.codegen import graph

_TEMPLATE_DIR = pathlib.Path(__file__).parent / 'templates'
_HEADER_TEMPLATE = _TEMPLATE_DIR / 'inference.h.mako'
_SOURCE_TEMPLATE = _TEMPLATE_DIR / 'inference.cc.mako'


class ModelData(TypedDict):
  header_file: str
  model_name: str
  op_code_table: graph.OpCodeTable
  graph: graph.Graph


def _render(output_file: pathlib.Path, template_file: pathlib.Path,
            model_data: ModelData) -> None:
  print("Generating {}".format(output_file))
  t = template.Template(filename=str(template_file))
  with output_file.open('w+') as file:
    file.write(t.render(**model_data))


def _generate_header(header_path: pathlib.Path, model_data: ModelData) -> None:
  _render(header_path, _HEADER_TEMPLATE, model_data)


def _generate_source(source_path: pathlib.Path, model_data: ModelData) -> None:
  _render(source_path, _SOURCE_TEMPLATE, model_data)


def generate(output_dir: str, output_name: str,
             op_code_table: graph.OpCodeTable, graph: graph.Graph) -> None:
  """ Generate C/C++ inference code. """
  header_file = f"{output_name}.h"
  model_data: ModelData = {
      'header_file': header_file,
      'model_name': output_name,
      'op_code_table': op_code_table,
      'graph': graph,
  }

  # Ensure output directory exists
  output_path = pathlib.Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  _generate_header(output_path / header_file, model_data)
  _generate_source(output_path / f"{output_name}.cc", model_data)
