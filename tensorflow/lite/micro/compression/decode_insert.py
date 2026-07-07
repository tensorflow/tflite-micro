# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

import warnings
from collections import defaultdict
from dataclasses import dataclass

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
  encoded_data: bytes
  ancillary_data: bytes
  consumers: list[model_editor.Operator]
  is_output: bool


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


def _rewrite_encoded_tensor(
    tensor: model_editor.Tensor,
    encoded_data: bytes,
) -> None:
  """Rewrite a compressed tensor to hold encoded data.

  The original tensor contained uncompressed values with quantization. After
  compression, it holds packed indices (or other encoded form) as raw bytes.
  This function updates the tensor in place to reflect its new role.

  Args:
    tensor: The tensor to rewrite.
    encoded_data: The compressed/encoded data bytes.
  """
  tensor.shape = (len(encoded_data), )
  tensor.dtype = tflite.TensorType.UINT8
  tensor.quantization = None
  tensor.buffer.data = encoded_data


def insert_decode_operators(
    model: model_editor.Model,
    compression_results: dict[tuple[int, int], compressor.CompressionResult],
) -> None:
  """Insert DECODE operators for all compressed tensors.

  This function modifies the model in-place, inserting a DECODE operator
  before any operator that uses a compressed tensor as input, and appending
  a DECODE for compressed tensors listed as subgraph outputs.

  A separate DECODE is inserted before each consumer, rather than sharing one
  DECODE output among all consumers. This is required because the interpreter's
  alternate decompression memory resets its allocation offset for each DECODE's
  Prepare, causing all DECODE outputs to be allocated at the same address. If
  two consumers share one DECODE and another DECODE runs between them, the
  intervening DECODE overwrites the shared output, corrupting data for the
  second consumer.

  Conversely, all compressed tensors read by one consumer are decoded by a
  single DECODE. Values a consumer needs simultaneously must coexist, and
  only the outputs of a single DECODE do: the allocation reset happens
  between DECODE operators, not between the outputs of one.

  For each consumer of compressed tensors:
  1. Create an ancillary data tensor (DCM + type-specific data) for each
     compressed tensor the consumer reads
  2. Create an output tensor with the same shape/dtype as each decoded
     tensor
  3. Insert one DECODE operator immediately before the consumer
  4. Rewire the consumer to use the DECODE outputs

  A subgraph's output list is treated as one more consumer, one which reads
  its tensors only after the last operator runs: a calling operator (IF,
  WHILE) copies subgraph outputs when the subgraph returns, and the client
  reads model outputs after invocation. Compressed tensors in the output
  list are therefore decoded by a single DECODE appended after the last
  operator, and their output list entries are rewired to the decoded
  values, which no other DECODE runs late enough to overwrite.

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
    is_output = tensor in subgraph.outputs

    if not consumers and not is_output:
      warnings.warn(
          f"Compressed tensor {tensor.name!r} (subgraph {sg_idx}, "
          f"tensor {tensor_idx}) has no consumers and is not a subgraph "
          "output. No DECODE operator will be inserted.",
          stacklevel=2)
      continue

    info = _CompressedTensorInfo(
        subgraph_idx=sg_idx,
        tensor_idx=tensor_idx,
        tensor=tensor,
        encoded_data=result.encoded_data,
        ancillary_data=result.ancillary_data,
        consumers=consumers,
        is_output=is_output,
    )
    by_subgraph[sg_idx].append(info)

  # Process each subgraph
  for sg_idx, tensor_infos in by_subgraph.items():
    subgraph = model.subgraphs[sg_idx]

    # Cache ancillary tensors by original tensor to avoid duplicates. Each
    # DECODE needs its own output tensor, but ancillary data is identical for
    # all DECODEs of the same compressed tensor.
    ancillary_cache: dict[model_editor.Tensor, model_editor.Tensor] = {}

    # Track tensors to rewrite after all output tensors are created, since
    # _create_output_tensor reads the original tensor's shape/dtype/quantization.
    tensors_to_rewrite: dict[model_editor.Tensor, bytes] = {}

    def ancillary_for(info: _CompressedTensorInfo) -> model_editor.Tensor:
      """Reuse or create the ancillary tensor for info's tensor.

      First use also schedules the original tensor's rewrite to encoded data.
      """
      ancillary = ancillary_cache.get(info.tensor)
      if ancillary is None:
        ancillary = _create_ancillary_tensor(info.ancillary_data, info.tensor)
        subgraph.tensors.append(ancillary)
        ancillary_cache[info.tensor] = ancillary
        tensors_to_rewrite[info.tensor] = info.encoded_data
      return ancillary

    def build_decode(
        infos: list[_CompressedTensorInfo]
    ) -> tuple[model_editor.Operator, list[model_editor.Tensor]]:
      """Build one DECODE operator decoding all of infos' tensors.

      Returns the operator and its decoded output tensors, parallel to
      infos.
      """
      inputs = []
      outputs = []
      for info in infos:
        ancillary_tensor = ancillary_for(info)
        decoded = _create_output_tensor(info.tensor)
        subgraph.tensors.append(decoded)
        inputs.extend([info.tensor, ancillary_tensor])
        outputs.append(decoded)
      op = model_editor.Operator(
          opcode=tflite.BuiltinOperator.CUSTOM,
          custom_code=DECODE_CUSTOM_OP_NAME,
          inputs=inputs,
          outputs=outputs,
      )
      return op, outputs

    # Group compressed tensors by consumer, then handle consumers in
    # reverse position order so insertions don't invalidate positions
    by_consumer: dict[model_editor.Operator, list[_CompressedTensorInfo]] = {}
    for info in tensor_infos:
      for consumer in info.consumers:
        by_consumer.setdefault(consumer, []).append(info)

    for consumer in sorted(by_consumer,
                           key=subgraph.operators.index,
                           reverse=True):
      infos = by_consumer[consumer]
      decode_op, decoded_tensors = build_decode(infos)

      # Insert DECODE immediately before this consumer
      insert_pos = subgraph.operators.index(consumer)
      subgraph.operators.insert(insert_pos, decode_op)

      # Rewire only this consumer to use the decoded outputs
      for info, decoded in zip(infos, decoded_tensors):
        _rewire_consumers([consumer], info.tensor, decoded)

    # Decode compressed tensors read from the subgraph's output list, all
    # with one DECODE appended after the last operator (see docstring).
    output_infos = [info for info in tensor_infos if info.is_output]
    if output_infos:
      decode_op, decoded_tensors = build_decode(output_infos)
      subgraph.operators.append(decode_op)
      for info, decoded in zip(output_infos, decoded_tensors):
        subgraph.outputs = [
            decoded if t is info.tensor else t for t in subgraph.outputs
        ]

    # Rewrite encoded tensors after all output tensors are created
    for tensor, encoded_data in tensors_to_rewrite.items():
      _rewrite_encoded_tensor(tensor, encoded_data)
