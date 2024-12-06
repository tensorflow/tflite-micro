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
#
"""Compression specifications.

This module provides tools for specifying how a model should be compressed,
notably, a method for reading such a specification in YAML, e.g., from a file.
Such 'specfiles' are written during model development to specify which tensors
should be compressed, by what method, and according to what parameters.
specfiles are read by the compression tool. They are not used by the TFLM
interpreter.
"""

from dataclasses import dataclass
import yaml

EXAMPLE_YAML_SPEC = """
tensors:

  - subgraph: 0
    tensor: 42
    compression:
      - lut:
          index_bitwidth: 4

  - subgraph: 0
    tensor: 55
    compression:
      - lut:
          index_bitwidth: 2

""" # This example is checked in this module's unit test.


class CompressionMethod:
  pass


@dataclass
class Tensor:
  "A compression specification for the indicated tensor."

  subgraph: int
  tensor: int
  compression: list[CompressionMethod]


@dataclass
class LookUpTableCompression(CompressionMethod):

  index_bitwidth: int


class ParseError(Exception):
  "Raised when the spec string cannot be parsed."

  def __init__(self, message="error parsing spec", wrapped_exception=None):
    super().__init__(f"{message}: {str(wrapped_exception)}")
    self.original_exception = wrapped_exception


def parse_yaml(y: str) -> list[Tensor]:
  "Parses a compression spec in a YAML string into its Python representation."
  try:
    config = yaml.safe_load(y)

    tensors = []
    for item in config["tensors"]:
      bitwidth = item["compression"][0]["lut"]["index_bitwidth"]
      tensor = Tensor(subgraph=item["subgraph"],
                      tensor=item["tensor"],
                      compression=[
                          LookUpTableCompression(index_bitwidth=bitwidth),
                      ])
      tensors.append(tensor)

  except Exception as e:
    raise ParseError() from e

  return tensors
