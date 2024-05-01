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
# =============================================================================
"""Utils to lstm_test_data_generator.py that helps to generate the test data for lstm kernel (lstm_test_data.cc)"""

import numpy as np
from copy import deepcopy


def clip_range(vals, bit_width):
  """Mimic integer calculation.
  Clip the range of vals based on bit width.
  e.g., clip_range([300], 8) = [127] since int8 have range [-128, 127]
  Args:
      vals (np.array): float representation of the integer values
      bit_width (int): number of desired bits for vals
  Returns:
      np.array : clipped vals
  """
  # Numpy integer calculation does not do saturation. Implement here
  min_val = -2**(bit_width - 1)
  max_val = 2**(bit_width - 1) - 1
  if vals.max() > max_val or vals.min() < min_val:
    print(f"WARNING: integer overflow!")
  return np.clip(vals, min_val, max_val)


def quantize_data(data, scale, zero_point=0, bit_width=8):
  """Quantize the data to integer type with desired bit width.
  The quantized data is represented using float since integer calculation in
  numpy may differ from other implementations (e.g., no integer saturation
  protection in numpy)
  Args:
      data (np.array): float data
      scale (float): quantization scale of the data
      zero_point (integer): quantization zero point of the data
      bit_width (int): number of representative bits for vals
  Returns:
      np.array : quantized data in float but clipped range
  """
  vals = np.round(data / scale) + zero_point
  return clip_range(vals, bit_width)


def dequantize_data(quantized_data, scale, zero_point=0):
  """Dequantize the data to integer type with desired bit width.
  Args:
      quantized_data (np.array): quantized data
      scale (float): quantization scale of the data
      zero_point (integer): quantization zero point of the data
  Returns:
      np.array : dequantized data
  """
  return scale * (quantized_data - zero_point)


def rescale(data, effective_scale, zero_point, num_bits):
  """Rescale the data to the effective scale """
  # q = r/s + z
  rescaled = np.round(data * effective_scale) + zero_point
  return clip_range(rescaled, num_bits)


def calculate_scale(min_val, max_val, num_bits=8, symmetry=False):
  """Calculate quantization scale from the range and bit width"""
  num_bins = np.power(2, num_bits) - 1
  if symmetry:
    return max(abs(min_val), abs(max_val)) / int(num_bins / 2)
  return np.array((max_val - min_val) / num_bins, dtype=np.float32)


def calculate_zp(min_val, scale, num_bits=8):
  """Calculate the zero point from the minimal value"""
  quantized_floor = -np.power(2, num_bits) / 2
  return int(quantized_floor - min_val / scale)


def sigmoid(x):
  """Sigmoid (floating point)"""
  return 1 / (1 + np.exp(-x))


def quantized_sigmoid(input, input_scale, output_scale, num_bits=16):
  """Sigmoid (integer)"""
  float_input = input * input_scale
  float_result = sigmoid(float_input)
  return quantize_data(float_result, output_scale, bit_width=num_bits)


def quantized_tanh(input, input_scale, output_scale, num_bits=16):
  """Tanh (integer)"""
  float_input = input * input_scale
  float_result = np.tanh(float_input)
  return quantize_data(float_result, output_scale, bit_width=num_bits)


class QuantizedTensor:
  """Data structure for a quantized tensor"""

  def __init__(self, float_data, scale, zero_point, symmetry, num_bits=8):
    """Tensor is initialized using the floating point data"""
    self.float_data = float_data
    self.scale = scale
    self.zero_point = int(zero_point)
    self.symmetry = symmetry
    self.num_bits = num_bits
    self.quantized_data = quantize_data(float_data, scale, zero_point,
                                        num_bits)

  @property
  def dequantized_data(self):
    """Dequantize the quantized tensor data back to floating point"""
    return dequantize_data(self.quantized_data, self.scale,
                           self.zero_point).flatten()


class QuantizedGateParams:
  """Hold the quantization data and corresponding information for a LSTM gate (forget/input/cell/output gate) """

  def __init__(
      self,
      quantized_activation_weight,
      quantized_recurrent_weight,
      bias_data_float,
      shape_info,
      bias_num_bits=32,
      cell_num_bits=16,
      modulation=False,
  ):
    self.shape_info = shape_info
    self.activation_weight = quantized_activation_weight
    self.recurrent_weight = quantized_recurrent_weight
    self.bias_data_float = bias_data_float
    self.modulation = modulation
    self.bias_num_bits = bias_num_bits
    self.cell_num_bits = cell_num_bits
    # For INT16 cell state, the input scale is Q3.12
    self.nonlinear_input_scale = np.power(2.0, -(cell_num_bits - 4))
    # For INT16 cell state, the output scale is Q0.15
    self.nonlinear_output_scale = np.power(2.0, -(cell_num_bits - 1))

  def quantize_bias_data(self, input_scale):
    bias_scale = self.activation_weight.scale * input_scale
    return quantize_data(self.bias_data_float, bias_scale, 0,
                         self.bias_num_bits)

  def fold_zeropoint(self, weight, zero_point):
    # W*real = W*(quant-zero_pt) = Wquant - Wzero_pt
    # Wzero_pt is precomputed here as a constant (implemented in TFLM)
    zp_vector = zero_point * np.ones(shape=(self.shape_info['input_dim'], 1))
    zero_folded_vector = np.dot(weight, zp_vector)
    return -1 * clip_range(zero_folded_vector, self.bias_num_bits)

  def compute_activation_bias(self, input_scale, input_zp):
    # Wz is precomputed here and added it to the original bias (same scale)
    zero_folded_vector = self.fold_zeropoint(
        self.activation_weight.quantized_data, input_zp)
    quantized_bias = self.quantize_bias_data(input_scale)
    return zero_folded_vector + quantized_bias

  def compute_recurrent_bias(self, recurrent_zp):
    # Wz is precomputed here
    return self.fold_zeropoint(self.recurrent_weight.quantized_data,
                               recurrent_zp)

  def effective_activation_scale(self, input_scale):
    # Combine input scale with output scale. Used for fc calculation
    return (self.activation_weight.scale * input_scale /
            self.nonlinear_input_scale)

  def effective_recurrence_scale(self, recurrent_scale):
    # Combine input scale with output scale. Used for fc calculation
    return (self.recurrent_weight.scale * recurrent_scale /
            self.nonlinear_input_scale)


def assemble_quantized_tensor(float_data,
                              min_val,
                              max_val,
                              symmetry,
                              num_bits=8):
  """Create a QuantizedTensor using floating point data, range information, and bit width"""
  scale = calculate_scale(min_val, max_val, num_bits, symmetry)
  zp = 0
  if not symmetry:
    zp = calculate_zp(min_val, scale, num_bits)
  return QuantizedTensor(float_data,
                         scale,
                         zp,
                         symmetry=symmetry,
                         num_bits=num_bits)


def create_gate_params(gate_parameters, model_config, modulation=False):
  """Create a QuantizedGateParams using the gate paramater information and the model configuration"""
  shape_info = model_config['shape_info']
  quantization_settings = model_config['quantization_settings']

  activation_weight_data = np.array(
      gate_parameters['activation_weight_data']).reshape(
          (shape_info['input_dim'], shape_info['state_dim']))
  activation_weight = assemble_quantized_tensor(
      activation_weight_data,
      activation_weight_data.min(),
      activation_weight_data.max(),
      True,
      quantization_settings['weight_bits'],
  )

  recurrent_weight_data = np.array(
      gate_parameters['recurrent_weight_data']).reshape(
          (shape_info['input_dim'], shape_info['state_dim']))

  recurrent_weight = assemble_quantized_tensor(
      recurrent_weight_data,
      recurrent_weight_data.min(),
      recurrent_weight_data.max(),
      True,
      quantization_settings['weight_bits'],
  )

  bias_data_float = np.array(gate_parameters['bias_data']).reshape(
      (shape_info['input_dim'], 1))
  gate_params = QuantizedGateParams(
      activation_weight,
      recurrent_weight,
      bias_data_float,
      shape_info,
      bias_num_bits=quantization_settings['bias_bits'],
      cell_num_bits=quantization_settings['cell_bits'],
      modulation=modulation,
  )
  return gate_params


def gate_calculation(input, hidden_state, gate_params, debug=False):
  """
  A gate calculation is tanh(FC(activation, activation weight) + FC(recurrent, recurrent weight)). 
  For modulation gate, sigmoid is used instead of tanh.

  Note: for debugging purpose, floating point calculation is conducted in parallel with the integer calculation
  """
  # Quantized Version
  input_fc = np.dot(gate_params.activation_weight.quantized_data,
                    input.quantized_data)
  input_fc += gate_params.compute_activation_bias(input.scale,
                                                  input.zero_point)
  input_fc = rescale(input_fc,
                     gate_params.effective_activation_scale(input.scale), 0,
                     gate_params.cell_num_bits)
  recurrent_fc = np.dot(gate_params.recurrent_weight.quantized_data,
                        hidden_state.quantized_data)
  recurrent_fc += gate_params.compute_recurrent_bias(hidden_state.zero_point)
  recurrent_fc = rescale(
      recurrent_fc, gate_params.effective_recurrence_scale(hidden_state.scale),
      0, gate_params.cell_num_bits)

  before_activation = clip_range(input_fc + recurrent_fc,
                                 gate_params.cell_num_bits)

  # Float Version
  float_result = np.dot(gate_params.activation_weight.float_data,
                        input.float_data)
  float_result += np.dot(gate_params.recurrent_weight.float_data,
                         hidden_state.float_data)
  float_result += gate_params.bias_data_float

  if debug:
    print(f'input fc: {input_fc.flatten()}')
    print(f'recurrent fc: {recurrent_fc.flatten()}')

    dequantized_res = dequantize_data(before_activation,
                                      gate_params.nonlinear_input_scale)
    print(f'Intermediate before activation: {before_activation.flatten()}')
    print(f'dequantized :{dequantized_res.flatten()} ')
    print(f'float computation result: {float_result.flatten()} ')

    diff = dequantized_res - float_result
    print(f'diff percentage (%): {abs(diff/float_result).flatten()*100}')

  if gate_params.modulation:
    activated = quantized_tanh(before_activation,
                               gate_params.nonlinear_input_scale,
                               gate_params.nonlinear_output_scale,
                               gate_params.cell_num_bits)
    float_result = np.tanh(float_result)
  else:
    activated = quantized_sigmoid(before_activation,
                                  gate_params.nonlinear_input_scale,
                                  gate_params.nonlinear_output_scale,
                                  gate_params.cell_num_bits)
    float_result = sigmoid(float_result)

  if debug:
    dequantized_res = dequantize_data(activated,
                                      gate_params.nonlinear_output_scale)
    print(f'Gate result: {activated.flatten()} ')
    print(f'Dequantized: {dequantized_res.flatten()} ')
    print(f'float computation result: {float_result.flatten()} ')
    diff = dequantized_res - float_result
    print(f'diff percentage (%): {abs(diff/float_result).flatten()*100}')

  return activated, float_result


# The LSTM class
class QuantizedLSTMDebugger(object):
  """Help the debugging process of the LSTM kernel implementation by
  1. Exposing the kernel internal computation 
  2. Run floating point calculation in parallel with the integer version
  """

  def __init__(
      self,
      kernel_config,
      kernel_params,
      init_hidden_state_vals,
      hiddens_state_range,
      init_cell_state_vals,
      cell_state_range,
      cell_clip=8,
  ):
    self.kernel_config = kernel_config
    self.forget_gate_params = create_gate_params(
        kernel_params['forget_gate_data'], kernel_config)
    self.input_gate_params = create_gate_params(
        kernel_params['input_gate_data'], kernel_config)
    self.modulation_gate_params = create_gate_params(
        kernel_params['cell_gate_data'], kernel_config, modulation=True)
    self.output_gate_params = create_gate_params(
        kernel_params['output_gate_data'], kernel_config)
    self.quantization_settings = kernel_config['quantization_settings']

    self.hidden_state_tensor = assemble_quantized_tensor(
        np.array(init_hidden_state_vals).reshape((-1, 1)),
        hiddens_state_range[0],
        hiddens_state_range[1],
        False,
        self.quantization_settings['activation_bits'],
    )
    self.cell_state_tensor = assemble_quantized_tensor(
        np.array(init_cell_state_vals).reshape((-1, 1)),
        cell_state_range[0],
        cell_state_range[1],
        True,
        self.quantization_settings['cell_bits'],
    )

    self.quantized_cell_clip = quantize_data(
        cell_clip,
        self.cell_state_tensor.scale,
        self.cell_state_tensor.zero_point,
        self.quantization_settings['cell_bits'],
    )

  def invoke(self, input_tensor, debug=False):
    assert (
        input_tensor.num_bits == self.quantization_settings['activation_bits'])

    prev_hidden_state_tensor = deepcopy(self.hidden_state_tensor)
    prev_cell_state_tensor = deepcopy(self.cell_state_tensor)

    prev_hidden_state_float = prev_hidden_state_tensor.float_data
    prev_cell_state_float = prev_cell_state_tensor.float_data

    # forget gate
    forget_gate_quant, forget_gate_float = gate_calculation(
        input_tensor, prev_hidden_state_tensor, self.forget_gate_params)

    self.cell_state_tensor.quantized_data = rescale(
        prev_cell_state_tensor.quantized_data * forget_gate_quant,
        self.forget_gate_params.nonlinear_output_scale,
        0,
        self.quantization_settings['cell_bits'],
    )
    self.cell_state_tensor.float_data = (prev_cell_state_float *
                                         forget_gate_float)

    # input gate
    input_gate_quant, input_gate_float = gate_calculation(
        input_tensor, prev_hidden_state_tensor, self.input_gate_params)

    modulation_gate_quant, modulation_gate_float = gate_calculation(
        input_tensor, prev_hidden_state_tensor, self.modulation_gate_params)

    gated_input_quant = rescale(
        input_gate_quant * modulation_gate_quant,
        self._calculate_effective_cell_scale(),
        0,
        self.quantization_settings['cell_bits'],
    )
    gated_input_float = input_gate_float * modulation_gate_float

    if (
        debug
    ):  # Hidden/cell state will be updated, break up the debug to record the intermediate state
      print('======================One Step LSTM======================')
      print('###### Forget Gate Output: ######')
      print(f'Quantized: {forget_gate_quant.flatten()}')
      dequantized_val = dequantize_data(
          forget_gate_quant, self.forget_gate_params.nonlinear_output_scale, 0)
      print(f'Dequantized : {dequantized_val.flatten()}')
      print(f'Float : {forget_gate_float.flatten()}')

      print('###### Cell state after forgetting: ######')
      print(f'Quantized: {self.cell_state_tensor.quantized_data.flatten()}')
      print(
          f'Dequantized: {self.cell_state_tensor.dequantized_data.flatten()}')
      print(f'Float : {self.cell_state_tensor.float_data.flatten()}')

      print('###### Input gate output: ######')
      print(f'Quantized: {input_gate_quant.flatten()}')
      dequantized_val = dequantize_data(
          input_gate_quant, self.input_gate_params.nonlinear_output_scale, 0)
      print(f'Dequantized: {dequantized_val.flatten()}')
      print(f'Float : {input_gate_float.flatten()}')

      print('###### cell gate output: ######')
      print(f'Quantized: {modulation_gate_quant.flatten()}')
      dequantized_val = dequantize_data(
          modulation_gate_quant,
          self.modulation_gate_params.nonlinear_output_scale,
          0,
      )
      print(f'Dequantized: {dequantized_val.flatten()}')
      print(f'Float : {modulation_gate_float.flatten()}')

      print('###### Gated input (input_gate * cell_gate): ######')
      print(f'Quantized: {gated_input_quant.flatten()}')
      dequantized_val = dequantize_data(gated_input_quant,
                                        self.cell_state_tensor.scale, 0)
      print(f'Dequantized: {dequantized_val.flatten()}')
      print(f'Float : {gated_input_float.flatten()}')

    # Update the cell state
    self.cell_state_tensor.quantized_data += gated_input_quant
    self._apply_cell_clip()
    self.cell_state_tensor.float_data += gated_input_float

    # output gate
    output_gate_quant, output_gate_float = gate_calculation(
        input_tensor, prev_hidden_state_tensor, self.output_gate_params)

    # Update the hidden state
    transformed_cell_quant = quantized_tanh(
        self.cell_state_tensor.quantized_data,
        self.output_gate_params.nonlinear_input_scale,
        self.output_gate_params.nonlinear_output_scale,
        self.cell_state_tensor.num_bits,
    )

    transformed_cell_float = np.tanh(self.cell_state_tensor.float_data)

    gated_output_quant = rescale(
        output_gate_quant * transformed_cell_quant,
        self._calculate_effective_output_scale(),
        self.hidden_state_tensor.zero_point,
        self.hidden_state_tensor.num_bits,
    )
    gated_output_float = output_gate_float * transformed_cell_float

    self.hidden_state_tensor.quantized_data = gated_output_quant
    self.hidden_state_tensor.float_data = gated_output_float

    if debug:
      print('###### Updated cell state): ######')
      print(f'Quantized: {self.cell_state_tensor.quantized_data.flatten()}')
      print(
          f'Dequantized: {self.cell_state_tensor.dequantized_data.flatten()}')
      print(f'Float : {self.cell_state_tensor.float_data.flatten()}')

      print('###### Output gate: ######')
      print(f'Quantized : {output_gate_quant.flatten()}')
      dequantized_val = dequantize_data(
          output_gate_quant, self.output_gate_params.nonlinear_output_scale, 0)
      print(f'Dequantized: {dequantized_val.flatten()}')
      print(f'Float : {output_gate_float.flatten()}')

      print('###### Tanh transformed cell: ######')
      print(f'Quantized: {transformed_cell_quant.flatten()}')
      dequantized_val = dequantize_data(
          transformed_cell_quant,
          self.output_gate_params.nonlinear_output_scale,
          0,
      )
      print(f'Dequantized: {dequantized_val.flatten()}')
      print(f'Float : {transformed_cell_float.flatten()}')

      print('###### Updated hidden state: ######')
      print(f'Quantized: {gated_output_quant.flatten()}')
      print(
          f'Dequantized: {self.hidden_state_tensor.dequantized_data.flatten()}'
      )
      print(f'Float : {gated_output_float.flatten()}')

      diff = abs(self.hidden_state_tensor.dequantized_data -
                 gated_output_float.flatten())
      max_diff_perc = diff / gated_output_float.flatten() * 100
      print(f'Max diff perc (%): {max_diff_perc}')
    return gated_output_quant, gated_output_float

  def _calculate_effective_output_scale(self):
    return (self.output_gate_params.nonlinear_output_scale *
            self.modulation_gate_params.nonlinear_output_scale /
            self.hidden_state_tensor.scale)

  def _calculate_effective_cell_scale(self):
    return (self.input_gate_params.nonlinear_output_scale *
            self.modulation_gate_params.nonlinear_output_scale /
            self.cell_state_tensor.scale)

  def _apply_cell_clip(self):
    cell_vals = self.cell_state_tensor.quantized_data
    if (cell_vals.max() > self.quantized_cell_clip
        or cell_vals.min() < -self.quantized_cell_clip):
      print(f'WARNING: cell values clip to {self.quantized_cell_clip}!')

    self.cell_state_tensor.quantized_data = np.round(
        np.clip(cell_vals, -self.quantized_cell_clip,
                self.quantized_cell_clip))
