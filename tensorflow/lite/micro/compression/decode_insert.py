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
"""DECODE operator insertion into TFLite model graphs.

This module inserts DECODE operators into a compressed model. DECODE operators
transform encoded tensors (with their paired ancillary data tensors) into
tensors ready for use by downstream operators.

The DECODE operator is registered as a custom operator named "TFLM_DECODE".
Each DECODE output requires two inputs: the encoded tensor and the ancillary
data tensor (containing the DCM header and decode-type-specific data).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

# Custom operator name for DECODE
DECODE_CUSTOM_OP_NAME = "TFLM_DECODE"


@dataclass
class _CompressedTensorInfo:
  """Information about a compressed tensor for DECODE insertion."""
  subgraph_idx: int
  tensor_idx: int
  tensor: model_editor.Tensor
  ancillary_data: bytes
  consumers: list[model_editor.Operator]


def _find_tensor_consumers(
    subgraph: model_editor.Subgraph,
    tensor: model_editor.Tensor,
) -> list[model_editor.Operator]:
  """Find all operators in subgraph that use tensor as an input."""
  consumers = []
  for op in subgraph.operators:
    if tensor in op.inputs:
      consumers.append(op)
  return consumers


def _find_earliest_consumer_position(
    subgraph: model_editor.Subgraph,
    consumers: list[model_editor.Operator],
) -> int:
  """Find the position of the earliest consumer in operator list."""
  min_pos = len(subgraph.operators)
  for consumer in consumers:
    try:
      pos = subgraph.operators.index(consumer)
      min_pos = min(min_pos, pos)
    except ValueError:
      pass
  return min_pos


def _create_ancillary_tensor(
    ancillary_data: bytes,
    original_tensor: model_editor.Tensor,
) -> model_editor.Tensor:
  """Create an ancillary data tensor for a compressed tensor.

  Args:
    ancillary_data: The complete ancillary data (DCM + type-specific data).
    original_tensor: The original tensor being decoded, for naming.

  Returns:
    A new Tensor containing the ancillary data.
  """
  name = None
  if original_tensor.name:
    name = f"{original_tensor.name}_ancillary"

  return model_editor.Tensor(
      shape=(len(ancillary_data), ),
      dtype=tflite.TensorType.UINT8,
      data=ancillary_data,
      name=name,
  )


def _create_output_tensor(
    original_tensor: model_editor.Tensor, ) -> model_editor.Tensor:
  """Create the output tensor for a DECODE operator.

  The output tensor has the same shape, dtype, and quantization as the
  original tensor would have when decoded. It has no data---the DECODE
  operator produces its values at runtime.

  Args:
    original_tensor: The original tensor being decoded.

  Returns:
    A new Tensor for the DECODE output.
  """
  name = None
  if original_tensor.name:
    name = f"{original_tensor.name}_decoded"

  return model_editor.Tensor(
      shape=original_tensor.shape,
      dtype=original_tensor.dtype,
      quantization=original_tensor.quantization,
      name=name,
  )


def _rewire_consumers(
    consumers: list[model_editor.Operator],
    old_tensor: model_editor.Tensor,
    new_tensor: model_editor.Tensor,
) -> None:
  """Replace old_tensor with new_tensor in all consumer inputs."""
  for consumer in consumers:
    consumer.inputs = [
        new_tensor if t is old_tensor else t for t in consumer.inputs
    ]


def insert_decode_operators(
    model: model_editor.Model,
    compression_results: dict[tuple[int, int], compressor.CompressionResult],
) -> None:
  """Insert DECODE operators for all compressed tensors.

  This function modifies the model in-place, inserting DECODE operators
  before any operator that uses a compressed tensor as input.

  A separate DECODE is inserted before each consumer, rather than sharing one
  DECODE output among all consumers. This is required because the interpreter's
  alternate decompression memory resets its allocation offset for each DECODE's
  Prepare, causing all DECODE outputs to be allocated at the same address. If
  two consumers share one DECODE and another DECODE runs between them, the
  intervening DECODE overwrites the shared output, corrupting data for the
  second consumer.

  For each consumer of a compressed tensor:
  1. Create an ancillary data tensor containing DCM + type-specific data
  2. Create an output tensor with the same shape/dtype as the decoded tensor
  3. Insert a DECODE operator immediately before the consumer
  4. Rewire the consumer to use the DECODE output

  Args:
    model: The model to modify in-place.
    compression_results: Map from (subgraph_idx, tensor_idx) to the
                         CompressionResult containing ancillary_data.
  """
  # Group compressed tensors by subgraph
  by_subgraph: dict[int, list[_CompressedTensorInfo]] = defaultdict(list)

  for (sg_idx, tensor_idx), result in compression_results.items():
    subgraph = model.subgraphs[sg_idx]
    tensor = subgraph.tensors[tensor_idx]
    consumers = _find_tensor_consumers(subgraph, tensor)

    if not consumers:
      # Tensor not used as input anywhere---no DECODE needed
      continue

    info = _CompressedTensorInfo(
        subgraph_idx=sg_idx,
        tensor_idx=tensor_idx,
        tensor=tensor,
        ancillary_data=result.ancillary_data,
        consumers=consumers,
    )
    by_subgraph[sg_idx].append(info)

  # Process each subgraph
  for sg_idx, tensor_infos in by_subgraph.items():
    subgraph = model.subgraphs[sg_idx]

    # Collect all (consumer, tensor_info) pairs and sort by consumer position
    # in reverse order so insertions don't invalidate positions
    consumer_pairs = []
    for info in tensor_infos:
      for consumer in info.consumers:
        consumer_pairs.append((consumer, info))

    consumer_pairs.sort(
        key=lambda pair: subgraph.operators.index(pair[0]),
        reverse=True,
    )

    for consumer, info in consumer_pairs:
      # Create ancillary data tensor (one per DECODE)
      ancillary_tensor = _create_ancillary_tensor(
          info.ancillary_data,
          info.tensor,
      )
      subgraph.tensors.append(ancillary_tensor)

      # Create output tensor (one per DECODE)
      output_tensor = _create_output_tensor(info.tensor)
      subgraph.tensors.append(output_tensor)

      # Create DECODE operator
      decode_op = model_editor.Operator(
          opcode=tflite.BuiltinOperator.CUSTOM,
          custom_code=DECODE_CUSTOM_OP_NAME,
          inputs=[info.tensor, ancillary_tensor],
          outputs=[output_tensor],
      )

      # Insert DECODE immediately before this consumer
      insert_pos = subgraph.operators.index(consumer)
      subgraph.operators.insert(insert_pos, decode_op)

      # Rewire only this consumer to use the decoded output
      _rewire_consumers([consumer], info.tensor, output_tensor)
