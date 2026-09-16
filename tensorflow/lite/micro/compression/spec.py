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
from typing import Optional, Union
import yaml

EXAMPLE_YAML_SPEC = """
tensors:

  - subgraph: 0
    tensor: 42
    compression:
      - lut:
          index_bitwidth: 4
          per_channel:
            axis: 0

  - subgraph: 0
    tensor: 55
    compression:
      - lut:
          index_bitwidth: 2
          per_tensor:

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
class PerTensor:
  """One value table for the whole tensor."""


@dataclass
class PerChannel:
  """One value table per slice along an axis.

  Attributes:
    axis: The axis of the tensor's shape that gives the channel count.
  """
  axis: int


@dataclass
class LookUpTableCompression(CompressionMethod):
  """LUT compression using lookup tables.

  Attributes:
    index_bitwidth: Number of bits per index (1-7).
    mode: PerTensor or PerChannel. Exactly one is required per lut
      entry.
  """
  index_bitwidth: int
  mode: Optional[Union[PerTensor, PerChannel]] = None


@dataclass
class HuffmanCompression(CompressionMethod):
  """Huffman compression using Xtensa-format decode tables.

  Supported tensor types: INT8, INT16 only.
  """
  pass


@dataclass
class PruningCompression(CompressionMethod):
  """Pruning (sparsity) compression.

  Supported tensor types: All TFLM tensor types.
  """
  pass


class ParseError(Exception):
  "Raised when the spec string cannot be parsed."

  def __init__(self, message="error parsing spec", wrapped_exception=None):
    super().__init__(f"{message}: {str(wrapped_exception)}")
    self.original_exception = wrapped_exception


def _parse_lut(lut: dict) -> LookUpTableCompression:
  """Parse a lut compression entry from its YAML dict."""
  has_per_tensor = "per_tensor" in lut
  has_per_channel = "per_channel" in lut
  if has_per_tensor and has_per_channel:
    raise ParseError(
        "lut: per_tensor and per_channel are contradictory; give exactly one")
  if not has_per_tensor and not has_per_channel:
    raise ParseError("lut: one of per_tensor or per_channel is required")

  if has_per_tensor:
    if lut["per_tensor"] is not None:
      raise ParseError("lut: per_tensor takes no value")
    mode = PerTensor()
  else:
    per_channel = lut["per_channel"]
    if not isinstance(per_channel, dict) or "axis" not in per_channel:
      raise ParseError("lut: per_channel requires an axis")
    axis = per_channel["axis"]
    if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
      raise ParseError("lut: per_channel axis must be a non-negative integer")
    mode = PerChannel(axis=axis)

  return LookUpTableCompression(index_bitwidth=lut["index_bitwidth"],
                                mode=mode)


def _parse_compression_method(comp: dict) -> CompressionMethod:
  """Parse a single compression method from YAML dict."""
  if "lut" in comp:
    return _parse_lut(comp["lut"])
  elif "huffman" in comp:
    return HuffmanCompression()
  elif "pruning" in comp:
    return PruningCompression()
  else:
    raise ParseError(f"Unknown compression method: {list(comp.keys())}")


def parse_yaml(y: str) -> list[Tensor]:
  "Parses a compression spec in a YAML string into its Python representation."
  try:
    config = yaml.safe_load(y)

    tensors = []
    for item in config["tensors"]:
      methods = []
      for comp in item["compression"]:
        methods.append(_parse_compression_method(comp))

      tensor = Tensor(
          subgraph=item["subgraph"],
          tensor=item["tensor"],
          compression=methods,
      )
      tensors.append(tensor)

  except ParseError:
    raise
  except Exception as e:
    raise ParseError() from e

  return tensors
