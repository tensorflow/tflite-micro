# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Builder pattern for creating compression specifications programmatically.

This module provides a fluent API for building compression specs without
needing to write YAML strings.

Example usage:
    from tflite_micro.compression import SpecBuilder
    
    spec = (SpecBuilder()
        .add_tensor(subgraph=0, tensor=2)
            .with_lut(index_bitwidth=4)
        .add_tensor(subgraph=0, tensor=4)
            .with_lut(index_bitwidth=2)
        .build())
"""

from typing import List, Optional
from . import spec


class TensorBuilder:
  """Builder for individual tensor compression specifications."""

  def __init__(self, subgraph: int, tensor: int,
               parent_builder: 'SpecBuilder'):
    self.subgraph = subgraph
    self.tensor = tensor
    self.compression_methods: List[spec.CompressionMethod] = []
    self._parent = parent_builder

  def with_lut(self, index_bitwidth: int) -> 'SpecBuilder':
    """Add LUT compression to this tensor.
        
        Args:
            index_bitwidth: Number of bits for the LUT index (e.g., 4 for 16 values)
            
        Returns:
            The parent SpecBuilder for method chaining
        """
    self.compression_methods.append(
        spec.LookUpTableCompression(index_bitwidth=index_bitwidth))
    return self._parent

  def _build(self) -> spec.Tensor:
    """Build the Tensor specification object."""
    return spec.Tensor(subgraph=self.subgraph,
                       tensor=self.tensor,
                       compression=self.compression_methods)


class SpecBuilder:
  """Fluent builder for compression specifications."""

  def __init__(self):
    self._tensor_builders: List[TensorBuilder] = []
    self._current_tensor: Optional[TensorBuilder] = None

  def add_tensor(self, subgraph: int, tensor: int) -> TensorBuilder:
    """Add a tensor to be compressed.
        
        Args:
            subgraph: The subgraph index containing the tensor
            tensor: The tensor index within the subgraph
            
        Returns:
            A TensorBuilder for configuring compression methods
        """
    # Finalize any current tensor
    if self._current_tensor is not None:
      self._tensor_builders.append(self._current_tensor)

    # Create new tensor builder
    self._current_tensor = TensorBuilder(subgraph, tensor, self)
    return self._current_tensor

  def build(self) -> List[spec.Tensor]:
    """Build the final compression specification.
        
        Returns:
            A list of Tensor specifications ready for use with compress()
        """
    # Make sure to include the last tensor if there is one
    if self._current_tensor is not None:
      self._tensor_builders.append(self._current_tensor)
      self._current_tensor = None

    return [tb._build() for tb in self._tensor_builders]
