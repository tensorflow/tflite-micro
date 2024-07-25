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
"""Flatbuffer utility functions that are specific to TensorFLow Lite Micro.

This module has a collection of utility function that perform flatbuffer
manipulation that is unsupported in the TfLite converter but are still needed
for Micro-specific use cases.

Transformation functions breakdown:
go/tflm-flatbuffer-reduction-breakdown
"""

import numpy as np

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb
from tflite_micro.tensorflow.lite.python import schema_util
from tflite_micro.tensorflow.lite.micro.tools import tflite_flatbuffer_align_wrapper


def remove_extraneous_quantization_data(model):
  """Remove min/max quantization data and shrink zero point arrays from weight/bias tensors."""
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.quantization is not None:
        # Remove unused min and max arrays from all tensors.
        tensor.quantization.max = []
        tensor.quantization.min = []

        # ensure a zero point is present for this tensor (non-quantized models
        # have quantization info = None)
        if tensor.quantization.zeroPoint is None:
          continue

        # We are only looking at reducing repeated zero points for the case of
        # per-channel quantized tensor. So if the zero point is already a scalar
        # (i.e. per tensor quantization) then we can exit early.
        if len(tensor.quantization.zeroPoint) == 1:
          continue

        # Only weight/bias tensors (where tensor.buffer != None)
        # are per-channel quantized
        if tensor.buffer is None:
          continue

        # Only the first zero_point value is used in the per-channel quantized
        # kernel implementation, which is assumed to be 0 anyways (TFLM only
        # has support for symmetric quantization).
        if all(value == 0 for value in tensor.quantization.zeroPoint):
          tensor.quantization.zeroPoint = [tensor.quantization.zeroPoint[0]]
        else:
          raise ValueError("TFLM only supports zero_point==0")


def shorten_variable_shared_names(model):
  """Replaces shared names with a shorter string corresponding to a unique index."""
  unique_shared_names = []
  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      builtin_code = schema_util.get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex])
      if builtin_code == schema_fb.BuiltinOperator.VAR_HANDLE:
        shared_name = op.builtinOptions.sharedName
        if shared_name not in unique_shared_names:
          unique_shared_names.append(shared_name)
        op.builtinOptions.sharedName = str(
            unique_shared_names.index(shared_name))


def _remove_initialization_subgraph(model):
  """Removes the resource variable initialization subgraph entirely from the flatbuffer for additional memory savings."""
  # assumption is made that subgraph indexed=1 is the resource variable
  # initialization subgraph (subgraph containing only pairs of VAR_HANDLE
  # and ASSIGN_VARIABLE OPs)
  non_initialization_subgraphs_list = [model.subgraphs[0]
                                       ] + model.subgraphs[2:]

  # TODO(b/279035671): add more documentation for why this is needed.
  # Make sure there is a proper VAR_HANDLE, ASSIGN_VARIABLE pair to allocate
  # each resource variable

  # global (across model) dict to store each unique shared_name with a boolean
  # conditional (True in the case it has an VAR_HANDLE/ASSIGN_VARIABLE pair in
  # the same subgraph anywhere in the model)
  shared_name_to_allocated_pair = {}
  for subgraph in non_initialization_subgraphs_list:
    # dict local to each subgraph matching a resource_id tensor with the unique
    # resource variable shared_name (Tensor indices are specific to a subgraph)
    id_to_shared_name_pairs = {}
    for op in subgraph.operators:
      builtin_code = schema_util.get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex])

      if builtin_code == schema_fb.BuiltinOperator.VAR_HANDLE:
        shared_name = op.builtinOptions.sharedName
        shared_name_to_allocated_pair.setdefault(shared_name, False)
        resource_id_tensor = subgraph.tensors[op.outputs[0]]
        id_to_shared_name_pairs.setdefault(resource_id_tensor, shared_name)

      elif builtin_code == schema_fb.BuiltinOperator.ASSIGN_VARIABLE:
        resource_id_tensor = subgraph.tensors[op.inputs[0]]
        shared_name = id_to_shared_name_pairs.get(resource_id_tensor)
        shared_name_to_allocated_pair[shared_name] = True

  # We can not remove subgraph 1 if there are any resource variables that don't
  # have a VAR_HANDLE/ASSIGN_VARIABLE pair. This is due to the specifics of how
  # resource variable buffers are allocated in the TFLM runtime.
  # See b/279035671 for more details.
  if any(val == False for val in shared_name_to_allocated_pair.values()):
    return

  # In preparation for removing subgraph 1 (resource variable initialization
  # subgraph) from the flatbuffer, any subgraph indices used by other OPs will
  # need to be updated to reflect the new change of having one fewer subgraph.
  for subgraph in model.subgraphs[2:]:
    for op in subgraph.operators:
      builtin_code = schema_util.get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex])
      if builtin_code == schema_fb.BuiltinOperator.CALL_ONCE:
        op.builtinOptions.initSubgraphIndex -= 1
      if builtin_code == schema_fb.BuiltinOperator.IF:
        op.builtinOptions.thenSubgraphIndex -= 1
        op.builtinOptions.elseSubgraphIndex -= 1
      elif builtin_code == schema_fb.BuiltinOperator.WHILE:
        op.builtinOptions.condSubgraphIndex -= 1
        op.builtinOptions.bodySubgraphIndex -= 1

  # safe to remove subgraph 1 from the flatbuffer
  model.subgraphs = non_initialization_subgraphs_list


def _remove_call_once_op(model):
  """Removes CALL_ONCE op for the resource variable initialization subgraph."""
  updated_op_list = []
  for op in model.subgraphs[0].operators:
    is_call_once = (schema_util.get_builtin_code_from_operator_code(
        model.operatorCodes[op.opcodeIndex]) ==
                    schema_fb.BuiltinOperator.CALL_ONCE)

    if is_call_once and op.builtinOptions.initSubgraphIndex == 1:
      # We make the assumption that subgraph indexed 1 is the resource variable
      # initialization subgraph, and as a result of the transformations, we no
      # longer need to execute the subgraph during runtime.
      continue

    updated_op_list.append(op)

  model.subgraphs[0].operators = updated_op_list


def _zero_bias_buffer(model, buffer_idx, zero_point):
  # Only clear buffer if its all zero_points
  # Ensure buffers are still present, but empty. This prevents the memory
  # planner from allocating arrays for the ASSIGN_VARIABLE input tensors in
  # subgraph 1.
  buffer = model.buffers[buffer_idx]
  if buffer.data is None:
    buffer.data = []
    return
  if len(buffer.data) == 0:
    return

  # For now this assumes that zero_point is int8 and hence all the buffer
  # data is as well. future work should update this to check for tensor.type
  # match to numpy type to load the data properly.
  buffer_data = np.frombuffer(buffer.data, dtype=np.int8)
  if all(value == zero_point for value in buffer_data):
    buffer.data = []


def _zero_resource_buffers(model):
  """Zero out resource buffers.

  Ignores buffers which are used in multiple subgraphs (b/266017172).
  Args:
    model: The model to operate on, a schema_fb.ModelT object.

  Returns:
    multi_subgraph_resource_buffers: list of resource variable buffers that are
    in multiple subgraphs.
  """
  multi_subgraph_resource_buffers = []
  # Each element in subgraph_buffers is a set containing the buffer index to
  # all the corresponding tensors of that subgraph.
  subgraph_buffers = [set() for _ in range(len(model.subgraphs))]
  for i, buffer_set in enumerate(subgraph_buffers):
    for tensor in model.subgraphs[i].tensors:
      buffer_set.add(tensor.buffer)

  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      builtin_code = schema_util.get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex])
      if builtin_code == schema_fb.BuiltinOperator.ASSIGN_VARIABLE:
        tensor = subgraph.tensors[op.inputs[1]]
        buffer_idx = tensor.buffer
        # List of subgraphs that use the buffer corresponding to the Op tensor
        buffer_in_subgraph = [
            buffer_idx in buffer_set for buffer_set in subgraph_buffers
        ]
        # If the buffer was only in one subgraph, it implies that it is used
        # for initialization only, and can be replaced with an empty array.
        if buffer_in_subgraph.count(True) == 1:
          zero_point = 0
          if tensor.quantization.zeroPoint:
            zero_point = tensor.quantization.zeroPoint[0]
          _zero_bias_buffer(model, buffer_idx, zero_point)
        else:
          multi_subgraph_resource_buffers.append(buffer_idx)
  return multi_subgraph_resource_buffers


def clear_resource_variable_buffers(model):
  """Clear resource variable buffers, removes assocaited CALL_ONCE op, and the resource buffer initialization subgraph."""
  multi_subgraph_resource_buffers = _zero_resource_buffers(model)

  # * We are assuming the resource variable initializaiton subgraph index is 1.
  if len(model.subgraphs) == 1:
    return
  found_non_resource_var_op = False
  for op in model.subgraphs[1].operators:
    builtin_code = schema_util.get_builtin_code_from_operator_code(
        model.operatorCodes[op.opcodeIndex])
    if (builtin_code != schema_fb.BuiltinOperator.VAR_HANDLE
        and builtin_code != schema_fb.BuiltinOperator.ASSIGN_VARIABLE):
      found_non_resource_var_op = True
      break

  if found_non_resource_var_op:
    # since subgraph 1 has OPs other than those associated with initializing
    # resource variables, we can't make any additional changes to the flatbuffer
    return

  for tensor in model.subgraphs[1].tensors:
    buffer_idx = tensor.buffer
    if (tensor.type != schema_fb.TensorType.RESOURCE
        and buffer_idx not in multi_subgraph_resource_buffers
        and model.buffers[buffer_idx].data != []):
      # if the entire initialization subgraph has not been cleared, we cannot
      # make any additional changes to the flatbuffer
      return

  # remove resource variable initialization subgraph
  _remove_call_once_op(model)
  _remove_initialization_subgraph(model)


def _numpy_from_tensor_type(tensor_type_idx):
  """Gives the equivalent numpy dtype based on TensorType class (schema) number."""
  tensor_type_idx_to_numpy = {
      schema_fb.TensorType.FLOAT32: np.float32,
      schema_fb.TensorType.FLOAT16: np.float16,
      schema_fb.TensorType.INT32: np.int32,
      schema_fb.TensorType.UINT8: np.uint8,
      schema_fb.TensorType.INT64: np.int64,
      schema_fb.TensorType.STRING: np.bytes_,
      schema_fb.TensorType.BOOL: np.bool_,
      schema_fb.TensorType.INT16: np.int16,
      schema_fb.TensorType.COMPLEX64: np.complex64,
      schema_fb.TensorType.INT8: np.int8,
      schema_fb.TensorType.FLOAT64: np.float64,
      schema_fb.TensorType.COMPLEX128: np.complex128,
      schema_fb.TensorType.UINT64: np.uint64,
      schema_fb.TensorType.RESOURCE: "RESORCE",
      schema_fb.TensorType.VARIANT: "VARIANT",
      schema_fb.TensorType.UINT32: np.uint32,
      schema_fb.TensorType.UINT16: np.uint16,
      # INT4 is mapped to INT8, b/246806634
      schema_fb.TensorType.INT4: np.int8,
  }
  return tensor_type_idx_to_numpy.get(tensor_type_idx)


def _get_minmax_range_int(dtype):
  """Returns the minimum and maximum range for an INT dtype."""
  return np.iinfo(dtype).min, np.iinfo(dtype).max


def _get_minmax_range_float(model, input_tensor):
  """Returns the minimum and maximum range for a FLOAT input_tensor.

  Assumes only one subgraph.
  If the tensor has an associated QUANTIZE Op, uses the quantization information
  to determine a more accurate range for random values.

  Args:
    model: schema_fb.ModelT model object (tflite model)
    input_tensor: a FLOAT dtype schema_fb.TensorT input tensor which's range to
      return

  Returns:
    range_min, range_max: the min/max values the input could have. default to
    [0, 1]
  """
  if _numpy_from_tensor_type(input_tensor.type) != np.float32:
    return
  if not any(input_tensor == model.subgraphs[0].tensors[input_idx]
             for input_idx in model.subgraphs[0].inputs):
    return
  # get associated quantize tensor
  # if there are multiple FLOAT32 inputs that get quantized, we assume
  # that each has their own quantize op, since quantize.cc ensures that
  # NumInputs and NumOutput == 1.
  for op in model.subgraphs[0].operators:
    if (schema_util.get_builtin_code_from_operator_code(
        model.operatorCodes[op.opcodeIndex])
        == schema_fb.BuiltinOperator.QUANTIZE
        and input_tensor == model.subgraphs[0].tensors[op.inputs[0]]):
      # use quantized tensor information for a more accurate F32 range
      quant_tensor = model.subgraphs[0].tensors[op.outputs[0]]
      dtype = _numpy_from_tensor_type(quant_tensor.type)
      scale = quant_tensor.quantization.scale[0]
      zero_point = quant_tensor.quantization.zeroPoint[0]
      # We add 1 to q_min to more accurately represent symmetrical
      # quantization (for INT16)
      r_min = float(np.iinfo(dtype).min + 1 - zero_point) * scale
      r_max = float(np.iinfo(dtype).max - zero_point) * scale
      return r_min, r_max

  return 0, 1


def generate_random_input_data(model, input_tensor, random_number_generator):
  """Generates random input data based on the tensor parameters (data_type and related quantization information).

  Not all input types are supported. RuntimeError is raised on unsupported type.
  Assumes a single subgraph model.

  Args:
    model: a tflite schema ModelT object
    input_tensor: the TensorT object whose parameters are matched to generate
      the random values.
    random_number_generator: a numpy.random number generator to get random
      values from

  Returns:
    array of input_tensor.shape of random data according to the input dtype.

  Raises:
    RuntimeError: for unsupported dtypes of input tensor.
  """
  dtype = _numpy_from_tensor_type(input_tensor.type)

  if dtype in (np.int8, np.int16):
    range_min, range_max = _get_minmax_range_int(dtype)
    return random_number_generator.integers(
        low=range_min,
        high=range_max,
        size=input_tensor.shape,
        dtype=dtype,
    )
  elif dtype == np.float32:
    range_min, range_max = _get_minmax_range_float(model, input_tensor)
    return (range_max - range_min) * random_number_generator.random(
        input_tensor.shape, dtype=dtype) + range_min
  elif dtype == np.bool_:
    range_min, range_max = 0, 1
    return random_number_generator.integers(
        low=range_min,
        high=range_max,
        size=input_tensor.shape,
        dtype=np.int8,
    ).astype(bool)
  else:
    raise RuntimeError(
        "Unsupported data type for generating data for input tensor.")


def tflite_flatbuffer_align(input_model_path, output_model_path):
  tflite_flatbuffer_align_wrapper.align_tflite_model(input_model_path,
                                                     output_model_path)
