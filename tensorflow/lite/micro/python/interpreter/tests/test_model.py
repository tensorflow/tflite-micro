# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Create a sample model for testing purposes.

A simple quantized int8 model with single ADD operation.
"""

import flatbuffers
import numpy as np

from tflite_micro.tensorflow.lite.python import schema_py_generated as schema_fb

TFLITE_SCHEMA_VERSION = 3


def _generate_quantization_parameters(builder, rmin, rmax, qmin, qmax):
  """Tensor-wise quantization parameter generation
  """

  def caclulate_scale(rmin, rmax, qmin, qmax):
    return (rmax - rmin) / (qmax - qmin)

  def calculate_zero_point(rmin, rmax, qmin, qmax):
    """Zero point (zp) = int(qmin - rmin/s) or int(qmax - rmax/s);
    use the one with smallest rounding error
    """
    scale = caclulate_scale(rmin, rmax, qmin, qmax)
    zp_from_min = qmin - rmin / scale
    zp_from_min_error = abs(qmin) + abs(rmin / scale)
    zp_from_max = qmax - rmax / scale
    zp_from_max_error = abs(qmax) + abs(rmax / scale)
    zp = zp_from_min
    if zp_from_max_error < zp_from_min_error:
      zp = zp_from_max
    if zp < qmin:
      return qmin
    if zp > qmax:
      return qmax
    return round(zp)

  if rmin > 0:
    raise ValueError("The quantization range [rmin, rmax] must include 0")
    # Range info
  schema_fb.QuantizationParametersStartMinVector(builder, 1)
  builder.PrependFloat32(rmin)
  quant_min_offset = builder.EndVector(1)
  schema_fb.QuantizationParametersStartMaxVector(builder, 1)
  builder.PrependFloat32(rmax)
  quant_max_offset = builder.EndVector(1)
  # scale
  schema_fb.QuantizationParametersStartScaleVector(builder, 1)
  scale = caclulate_scale(rmin, rmax, qmin, qmax)
  builder.PrependFloat32(scale)
  quant_scale_offset = builder.EndVector(1)
  # zero point
  schema_fb.QuantizationParametersStartZeroPointVector(builder, 1)
  zp = calculate_zero_point(rmin, rmax, qmin, qmax)
  builder.PrependInt64(zp)
  quant_zero_point_offset = builder.EndVector(1)
  # put into the buffer
  schema_fb.QuantizationParametersStart(builder)
  schema_fb.QuantizationParametersAddMin(builder, quant_min_offset)
  schema_fb.QuantizationParametersAddMax(builder, quant_max_offset)
  schema_fb.QuantizationParametersAddScale(builder, quant_scale_offset)
  schema_fb.QuantizationParametersAddZeroPoint(builder,
                                               quant_zero_point_offset)
  quantization_offset = schema_fb.QuantizationParametersEnd(builder)
  return quantization_offset


def build_mock_flatbuffer_model():
  """Creates a flatbuffer containing an example model (ADD)."""
  builder = flatbuffers.Builder(1024)

  # Create buffers
  schema_fb.BufferStart(builder)
  # Buffer0 is used for the input data
  buffer0_offset = schema_fb.BufferEnd(builder)
  # Buffer 1 contains the constant to be added
  np_data = np.array([-50, -10, 20, 40], dtype=np.int8)
  data_bytes = np_data.tobytes()
  schema_fb.BufferStartDataVector(builder, len(data_bytes))
  for byte_data in data_bytes[::-1]:
    builder.PrependByte(byte_data)
  buffer1_data_offset = builder.EndVector(len(data_bytes))
  schema_fb.BufferStart(builder)
  schema_fb.BufferAddData(builder, buffer1_data_offset)
  buffer1_offset = schema_fb.BufferEnd(builder)
  # Buffer 2 is for the output tensor
  schema_fb.BufferStart(builder)
  buffer2_offset = schema_fb.BufferEnd(builder)
  # Add all the buffers to the fb
  schema_fb.ModelStartBuffersVector(builder, 3)
  builder.PrependUOffsetTRelative(buffer2_offset)
  builder.PrependUOffsetTRelative(buffer1_offset)
  builder.PrependUOffsetTRelative(buffer0_offset)
  buffers_offset = builder.EndVector(3)

  # Create tensors
  # Tensor0 (name: input_tensor, shape: (1,2,2), type: int8,  buffer_index: 0)
  string0_offset = builder.CreateString('input_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(2)
  builder.PrependInt32(2)
  builder.PrependInt32(1)
  shape0_offset = builder.EndVector(3)
  # Add Quantization parameters to the input tensor (tensor level quantization)
  # Float range: [-1,1]; Integer range: [-128, 127]
  # Scale : 2/255; Z: int(-128 + -1/2/255) = 0
  quantization0_offset = _generate_quantization_parameters(builder,
                                                           rmin=-1,
                                                           rmax=1,
                                                           qmin=-128,
                                                           qmax=127)
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string0_offset)
  schema_fb.TensorAddShape(builder, shape0_offset)
  schema_fb.TensorAddType(builder, 9)
  schema_fb.TensorAddBuffer(builder, 0)
  schema_fb.TensorAddQuantization(builder, quantization0_offset)
  tensor0_offset = schema_fb.TensorEnd(builder)
  # Tensor1 (name: constant_tensor, shape: (1,2,2), type: int8,  buffer_index: 1)
  string1_offset = builder.CreateString('constant_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(2)
  builder.PrependInt32(2)
  builder.PrependInt32(1)
  shape1_offset = builder.EndVector(3)
  # Add Quantization parameters to the constant tensor (tensor level quantization)
  # Float range: [0,1]; Integer range: [-128, 127]
  # Scale : 1/255; Z: int(-128 + 0/1/255) = -128
  quantization1_offset = _generate_quantization_parameters(builder,
                                                           rmin=0,
                                                           rmax=1,
                                                           qmin=-128,
                                                           qmax=127)
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string1_offset)
  schema_fb.TensorAddShape(builder, shape1_offset)
  schema_fb.TensorAddType(builder, 9)
  schema_fb.TensorAddBuffer(builder, 1)
  schema_fb.TensorAddQuantization(builder, quantization1_offset)
  tensor1_offset = schema_fb.TensorEnd(builder)
  # Tensor2 (name: output_tensor, shape: (1,2,2), type: int8,  buffer_index: 2)
  string2_offset = builder.CreateString('output_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(2)
  builder.PrependInt32(2)
  builder.PrependInt32(1)
  shape2_offset = builder.EndVector(3)
  # Add Quantization parameters to the output tensor (tensor level quantization)
  # Float range: [-1,2]; Integer range: [-128, 127]
  # Scale : 3/255; Z: int(-128 + -1/3/255) = -43
  quantization2_offset = _generate_quantization_parameters(builder,
                                                           rmin=-1,
                                                           rmax=2,
                                                           qmin=-128,
                                                           qmax=127)
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string2_offset)
  schema_fb.TensorAddShape(builder, shape2_offset)
  schema_fb.TensorAddType(builder, 9)
  schema_fb.TensorAddBuffer(builder, 2)
  schema_fb.TensorAddQuantization(builder, quantization2_offset)
  tensor2_offset = schema_fb.TensorEnd(builder)
  # Add all the tensors to the fb
  schema_fb.SubGraphStartTensorsVector(builder, 3)
  builder.PrependUOffsetTRelative(tensor2_offset)
  builder.PrependUOffsetTRelative(tensor1_offset)
  builder.PrependUOffsetTRelative(tensor0_offset)
  tensors_offset = builder.EndVector(3)

  # I/O for the subgraph
  # Tensor0 for input
  schema_fb.SubGraphStartInputsVector(builder, 1)
  builder.PrependInt32(0)
  inputs_offset = builder.EndVector(1)
  # Tensor2 for output
  schema_fb.SubGraphStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  outputs_offset = builder.EndVector(1)

  # The ADD operator
  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(builder, schema_fb.BuiltinOperator.ADD)
  schema_fb.OperatorCodeAddDeprecatedBuiltinCode(builder,
                                                 schema_fb.BuiltinOperator.ADD)
  schema_fb.OperatorCodeAddVersion(builder, 1)
  code_offset = schema_fb.OperatorCodeEnd(builder)

  schema_fb.ModelStartOperatorCodesVector(builder, 1)
  builder.PrependUOffsetTRelative(code_offset)
  codes_offset = builder.EndVector(1)
  # Tensor 0 and tensor 1 as the operands for the ADD operator
  schema_fb.OperatorStartInputsVector(builder, 2)
  builder.PrependInt32(0)
  builder.PrependInt32(1)
  op_inputs_offset = builder.EndVector(2)
  # Output is tensor 2
  schema_fb.OperatorStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  op_outputs_offset = builder.EndVector(1)
  # Add the ADD operator to fb
  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddOpcodeIndex(builder, 0)
  schema_fb.OperatorAddInputs(builder, op_inputs_offset)
  schema_fb.OperatorAddOutputs(builder, op_outputs_offset)
  op_offset = schema_fb.OperatorEnd(builder)
  # List of operators (only ADD in this case)
  schema_fb.SubGraphStartOperatorsVector(builder, 1)
  builder.PrependUOffsetTRelative(op_offset)
  ops_offset = builder.EndVector(1)

  # Start construct the subgraph
  string3_offset = builder.CreateString('subgraph_name')
  schema_fb.SubGraphStart(builder)
  schema_fb.SubGraphAddName(builder, string3_offset)
  schema_fb.SubGraphAddTensors(builder, tensors_offset)
  schema_fb.SubGraphAddInputs(builder, inputs_offset)
  schema_fb.SubGraphAddOutputs(builder, outputs_offset)
  schema_fb.SubGraphAddOperators(builder, ops_offset)
  subgraph_offset = schema_fb.SubGraphEnd(builder)
  # Only one subgraph
  schema_fb.ModelStartSubgraphsVector(builder, 1)
  builder.PrependUOffsetTRelative(subgraph_offset)
  subgraphs_offset = builder.EndVector(1)

  # Signature for the graph
  signature_key = builder.CreateString('my_key')
  input_tensor_string = builder.CreateString('input_tensor')
  output_tensor_string = builder.CreateString('output_tensor')
  # Signature Inputs
  schema_fb.TensorMapStart(builder)
  schema_fb.TensorMapAddName(builder, input_tensor_string)
  schema_fb.TensorMapAddTensorIndex(builder, 0)
  input_tensor = schema_fb.TensorMapEnd(builder)
  schema_fb.SignatureDefStartInputsVector(builder, 1)
  builder.PrependUOffsetTRelative(input_tensor)
  signature_inputs_offset = builder.EndVector(1)
  # Signature Outputs
  schema_fb.TensorMapStart(builder)
  schema_fb.TensorMapAddName(builder, output_tensor_string)
  schema_fb.TensorMapAddTensorIndex(builder, 2)
  output_tensor = schema_fb.TensorMapEnd(builder)
  schema_fb.SignatureDefStartOutputsVector(builder, 1)
  builder.PrependUOffsetTRelative(output_tensor)
  signature_outputs_offset = builder.EndVector(1)
  # Add schema to fb
  schema_fb.SignatureDefStart(builder)
  schema_fb.SignatureDefAddSignatureKey(builder, signature_key)
  schema_fb.SignatureDefAddInputs(builder, signature_inputs_offset)
  schema_fb.SignatureDefAddOutputs(builder, signature_outputs_offset)
  signature_offset = schema_fb.SignatureDefEnd(builder)
  schema_fb.ModelStartSignatureDefsVector(builder, 1)
  builder.PrependUOffsetTRelative(signature_offset)
  signature_defs_offset = builder.EndVector(1)

  # Start assembling the model
  string4_offset = builder.CreateString('model_description')
  schema_fb.ModelStart(builder)
  schema_fb.ModelAddVersion(builder, TFLITE_SCHEMA_VERSION)
  schema_fb.ModelAddOperatorCodes(builder, codes_offset)
  schema_fb.ModelAddSubgraphs(builder, subgraphs_offset)
  schema_fb.ModelAddDescription(builder, string4_offset)
  schema_fb.ModelAddBuffers(builder, buffers_offset)
  schema_fb.ModelAddSignatureDefs(builder, signature_defs_offset)
  model_offset = schema_fb.ModelEnd(builder)
  builder.Finish(model_offset)
  model = builder.Output()

  return model


def load_model_from_flatbuffer(flatbuffer_model):
  """Loads a model as a python object from a flatbuffer model."""
  model = schema_fb.Model.GetRootAsModel(flatbuffer_model, 0)
  model = schema_fb.ModelT.InitFromObj(model)
  return model


def build_mock_model():
  """Creates an object containing an example model."""
  model = build_mock_flatbuffer_model()
  return load_model_from_flatbuffer(model)
