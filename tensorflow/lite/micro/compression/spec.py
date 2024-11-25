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

from dataclasses import dataclass
import yaml


@dataclass
class Tensor:
  subgraph: int
  tensor: int
  compression: list


@dataclass
class LookUpTableCompression:
  index_bitwidth: int


EXAMPLE_YAML_SPEC = """\
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
          index_bitwidth: 2\
"""
# ^ This example is checked in this module's unit test.


def parse_yaml(y):
  config = yaml.safe_load(y)
  tensors = []

  for item in config["tensors"]:
    tensor = Tensor(
        subgraph=item["subgraph"],
        tensor=item["tensor"],
        compression=[
            LookUpTableCompression(index_bitwidth=item["compression"][0]["lut"]
                                   ["index_bitwidth"]),
        ],
    )
    tensors.append(tensor)

  return tensors
