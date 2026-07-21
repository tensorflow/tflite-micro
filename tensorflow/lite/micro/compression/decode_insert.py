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

  The output tensor is a copy of the original tensor, differing only in
  name and in having no data: the DECODE operator produces the values
  at runtime.

  Args:
    original_tensor: The original tensor being decoded.

  Returns:
    A new Tensor for the DECODE output.
  """
  name = None
  if original_tensor.name:
    name = f"{original_tensor.name}_decoded"

  tensor = original_tensor.copy(name=name)
  tensor.buffer = None
  return tensor


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
  The tensor receives a fresh Buffer, leaving the original buffer and any
  tensors aliasing it untouched; identical encodings converge again in the
  final deduplication pass.

  Args:
    tensor: The tensor to rewrite.
    encoded_data: The compressed/encoded data bytes.
  """
  tensor.shape = (len(encoded_data), )
  tensor.dtype = tflite.TensorType.UINT8
  tensor.quantization = None
  tensor.buffer = model_editor.Buffer(data=encoded_data)


def _drop_partially_covered_buffers(
    model: model_editor.Model,
    compression_results: dict[tuple[int, int], compressor.CompressionResult],
) -> dict[tuple[int, int], compressor.CompressionResult]:
  """Drop compressed tensors whose buffer an uncompressed tensor shares.

  The uncompressed tensor keeps the original data in the model, so
  compressing any alias of its buffer adds encoded data, ancillary
  data, and DECODE latency without reducing model size. Warn and
  return the results without the dropped entries.

  Args:
    model: The model the results apply to.
    compression_results: Map from (subgraph_idx, tensor_idx) to
                         CompressionResult.

  Returns:
    compression_results, minus entries for partially covered buffers.
  """
  coordinates = {
      id(model.subgraphs[s].tensors[t]): (s, t)
      for (s, t) in compression_results
  }
  by_buffer: dict[int, list[model_editor.Tensor]] = defaultdict(list)
  for tensor in model_editor.iter_tensors(model):
    if tensor.buffer is not None:
      by_buffer[id(tensor.buffer)].append(tensor)

  results = dict(compression_results)
  for aliases in by_buffer.values():
    covered = [t for t in aliases if id(t) in coordinates]
    if covered and len(covered) < len(aliases):
      uncovered = [t for t in aliases if id(t) not in coordinates]
      warnings.warn(
          f"Not compressing tensor(s) "
          f"{[t.name for t in covered]}: sharing a buffer with "
          f"uncompressed tensor(s) {[t.name for t in uncovered]}, whose "
          "data stays in the model, so compression cannot reduce model "
          "size.",
          stacklevel=3)
      for tensor in covered:
        del results[coordinates[id(tensor)]]
  return results


def insert_decode_operators(
    model: model_editor.Model,
    compression_results: dict[tuple[int, int], compressor.CompressionResult],
) -> None:
  """Insert DECODE operators for all compressed tensors.

  This function modifies the model in-place, inserting a DECODE operator
  before any operator that uses a compressed tensor as input, and appending
  a DECODE for compressed tensors listed as subgraph outputs.

  A separate DECODE is inserted before each consumer, and one DECODE
  decodes all the compressed tensors its consumer reads. DECODE outputs
  are tensors with a lifetime limited to the very next operator in the
  subgraph, so sharing one DECODE among multiple consumers would
  violate the lifetime rule. The DECODE operator trades increased
  latency for decreased memory usage.

  For each consumer of compressed tensors:
  1. Create an ancillary data tensor (DCM + type-specific data) for each
     compressed tensor the consumer reads
  2. Create an output tensor as a copy of each original tensor
  3. Insert one DECODE operator immediately before the consumer
  4. Rewire the consumer to use the DECODE outputs

  A subgraph's output list is treated as one more consumer, one which
  reads its tensors only after the last operator runs: a calling
  operator (IF, WHILE) copies subgraph outputs when the subgraph
  returns. Compressed tensors in the output list are therefore decoded
  by a single DECODE appended after the last operator, and their output
  list entries are rewired to the decoded values.

  Distinct tensors can share one buffer, in the same or different
  subgraphs, where the converter deduplicated identical constants.
  Compressed tensors sharing a buffer with uncompressed tensors are
  skipped with a warning: the uncompressed data must stay in the model,
  so compressing an alias cannot reduce model size. Otherwise each
  rewritten tensor and ancillary tensor receives its own buffer, and a
  final deduplication pass merges byte-identical buffers and prunes
  unreferenced ones, preserving the converter's sharing wherever
  compression results allow and dissolving it where they diverge.

  Args:
    model: The model to modify in-place.
    compression_results: Map from (subgraph_idx, tensor_idx) to the
                         CompressionResult containing ancillary_data.
  """
  compression_results = _drop_partially_covered_buffers(
      model, compression_results)

  # Group compressed tensors by subgraph
  by_subgraph: dict[int, list[_CompressedTensorInfo]] = defaultdict(list)

  for (sg_idx, tensor_idx), result in compression_results.items():
    subgraph = model.subgraphs[sg_idx]
    tensor = subgraph.tensors[tensor_idx]
    consumers = subgraph.consumers_of(tensor)
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

    # Cache ancillary tensors by content to avoid duplicates within
    # this subgraph. Each DECODE needs its own output tensor, but
    # DECODEs whose ancillary data coincides can read one tensor.
    ancillary_cache: dict[bytes, model_editor.Tensor] = {}

    # Track tensors to rewrite after all output tensors are created, since
    # _create_output_tensor reads the original tensor's shape/dtype/quantization.
    tensors_to_rewrite: dict[model_editor.Tensor, bytes] = {}

    def ancillary_for(info: _CompressedTensorInfo) -> model_editor.Tensor:
      """Reuse or create the ancillary tensor for info's ancillary data."""
      ancillary = ancillary_cache.get(info.ancillary_data)
      if ancillary is None:
        ancillary = _create_ancillary_tensor(info.ancillary_data, info.tensor)
        subgraph.tensors.append(ancillary)
        ancillary_cache[info.ancillary_data] = ancillary
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
        tensors_to_rewrite[info.tensor] = info.encoded_data
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

    # Positions of the original operators, computed once so the sort and
    # insertions below avoid a linear scan per lookup.
    op_position = {op: i for i, op in enumerate(subgraph.operators)}

    # Group compressed tensors by consumer, then handle consumers in
    # reverse position order so insertions don't invalidate positions:
    # each insertion falls after every consumer still to be processed,
    # leaving the recorded positions valid.
    by_consumer: dict[model_editor.Operator, list[_CompressedTensorInfo]] = {}
    for info in tensor_infos:
      for consumer in info.consumers:
        by_consumer.setdefault(consumer, []).append(info)

    for consumer in sorted(by_consumer,
                           key=lambda op: op_position[op],
                           reverse=True):
      infos = by_consumer[consumer]
      decode_op, decoded_tensors = build_decode(infos)

      # Insert DECODE immediately before this consumer
      subgraph.operators.insert(op_position[consumer], decode_op)

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

  # Every rewrite and ancillary tensor made a fresh buffer; converge
  # byte-identical ones and drop those left unreferenced, preserving
  # the sharing the converter created wherever results allow.
  model_editor.dedupe_buffers(model)
  model_editor.prune_buffers(model)
