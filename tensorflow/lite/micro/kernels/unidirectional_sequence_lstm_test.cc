/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/micro_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test_config.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)

constexpr int kLstmMaxNumInputTensors = 24;
constexpr int kLstmOutputTensorIndex = kLstmMaxNumInputTensors;
constexpr int kLstmIntermediateTensorBase = kLstmMaxNumInputTensors + 1;

template <typename T>
TfLiteTensor CreateQuantizedWeightTensor(const float* weights,
                                         T* quantized_weights,
                                         TfLiteIntArray* dims, int size,
                                         TfLiteFloatArray* scale,
                                         TfLiteIntArray* zero_point,
                                         TfLiteAffineQuantization* qparam) {
  TfLiteTensor tensor;
  float min;
  float max;
  float scaling_factor;
  micro_tensor_utils::SymmetricQuantizeFloats(
      weights, size, reinterpret_cast<int8_t*>(quantized_weights), &min, &max,
      &scaling_factor);
  tensor.dims = dims;
  tensor.is_variable = false;
  tensor.type = typeToTfLiteType<T>();
  tensor.data.data = const_cast<T*>(quantized_weights);
  tensor.bytes = ElementCount(*dims) * sizeof(T);
  tensor.quantization.type = kTfLiteAffineQuantization;
  tensor.params.scale = scaling_factor;
  tensor.params.zero_point = 0;
  scale->size = 1;
  scale->data[0] = scaling_factor;
  zero_point->size = 1;
  zero_point->data[0] = 0;
  qparam->quantized_dimension = 0;
  qparam->scale = scale;
  qparam->zero_point = zero_point;
  tensor.quantization.params = qparam;
  return tensor;
}

template <typename T>
QuantizationParams SetQuantizationParams(float f_min, float f_max) {
  QuantizationParams qparam;
  int32_t zero_point = 0;
  float scale = 0;
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const float qmin_float = static_cast<float>(qmin);
  const float qmax_float = static_cast<float>(qmax);
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_DCHECK_LE(f_min, 0);
  TFLITE_DCHECK_GE(f_max, 0);
  if (f_min == f_max) {
    // Special case where the min,max range is a point. Should be {0}.
    TFLITE_DCHECK_EQ(f_min, 0);
    TFLITE_DCHECK_EQ(f_max, 0);
    qparam.scale = static_cast<double>(scale);
    qparam.zero_point = zero_point;
    return qparam;
  }

  // General case.
  //
  // First determine the scale.
  scale = (f_max - f_min) / (qmax_float - qmin_float);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const float zero_point_from_min = qmin_float - f_min / scale;
  const float zero_point_from_max = qmax_float - f_max / scale;

  const float zero_point_from_min_error =
      std::abs(qmin_float) + std::abs(f_min / scale);

  const float zero_point_from_max_error =
      std::abs(qmax_float) + std::abs(f_max / scale);

  const float zero_point_float =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  //  padding).

  T nudged_zero_point = 0;
  if (zero_point_float < qmin_float) {
    nudged_zero_point = qmin;
  } else if (zero_point_float > qmax_float) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(round(zero_point_float));
  }

  // The zero point should always be in the range of quantized value,
  // // [qmin, qmax].
  TFLITE_DCHECK_GE(nudged_zero_point, qmin);
  TFLITE_DCHECK_LE(nudged_zero_point, qmax);

  zero_point = nudged_zero_point;
  // finally, return the values
  qparam.scale = static_cast<double>(scale);
  qparam.zero_point = zero_point;
  return qparam;
}

void CopyInputOrExpectedOutput(const float* input, float* dest,
                               bool batch_major, int sequence_length,
                               int n_batch, int n_input_output) {
  if (batch_major) {
    memcpy(dest, input,
           n_batch * sequence_length * n_input_output * sizeof(float));
  } else {
    for (int s = 0; s < sequence_length; ++s) {
      for (int b = 0; b < n_batch; ++b) {
        const float* batch_start =
            &input[b * sequence_length * n_input_output + s * n_input_output];
        memcpy(&dest[s * n_batch * n_input_output + b * n_input_output],
               batch_start, n_input_output * sizeof(float));
      }
    }
  }
}

void TestUnidirectionalSequenceLstmHybrid(
    LstmFloatTestConfig* config, LstmWeightQuantizationBuffers* quant,
    float tolerance, bool input_output_batch_major,

    TfLiteType weight_type,

    bool asymmetric_quantize_inputs = false) {
  int inputs_array_data[25];
  int outputs_array_data[2] = {1, kLstmOutputTensorIndex};

  if (config->use_layer_norm) {
    inputs_array_data[0] = 24;
  } else {
    inputs_array_data[0] = 20;
  }

  TfLiteTensor tensors[kLstmMaxNumInputTensors + 1];

  CopyInputOrExpectedOutput(config->input_original, config->input,
                            input_output_batch_major, config->sequence_length,
                            config->n_batch, config->n_input);

  int input_dim[4] = {3, config->sequence_length, config->n_batch,
                      config->n_input};
  tensors[kLstmInputTensor] =
      CreateTensor<float>(config->input, IntArrayFromInts(input_dim));
  inputs_array_data[kLstmInputTensor + 1] = kLstmInputTensor;

  int input_w_dim[3] = {2, config->n_cell, config->n_input};

  if (config->use_cifg) {
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    if (weight_type == kTfLiteUInt8) {
      tensors[kLstmInputToInputWeightsTensor] =
          CreateQuantizedWeightTensor<uint8_t>(
              config->input_to_input_weights,
              reinterpret_cast<uint8_t*>(quant->lstm_i2i_quant),
              IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
              FloatArrayFromFloats(quant->lstm_i2i_scale),
              IntArrayFromInts(quant->lstm_i2i_zp), quant->lstm_i2i_qparam);
      inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
          kLstmInputToInputWeightsTensor;
    } else if (weight_type == kTfLiteInt8) {
      tensors[kLstmInputToInputWeightsTensor] =
          CreateQuantizedWeightTensor<int8_t>(
              config->input_to_input_weights, quant->lstm_i2i_quant,
              IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
              FloatArrayFromFloats(quant->lstm_i2i_scale),
              IntArrayFromInts(quant->lstm_i2i_zp), quant->lstm_i2i_qparam);
      inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
          kLstmInputToInputWeightsTensor;
    } else {
      inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    }
  }

  if (weight_type == kTfLiteUInt8) {
    tensors[kLstmInputToForgetWeightsTensor] =
        CreateQuantizedWeightTensor<uint8_t>(
            config->input_to_forget_weights,
            reinterpret_cast<uint8_t*>(quant->lstm_i2f_quant),
            IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
            FloatArrayFromFloats(quant->lstm_i2f_scale),
            IntArrayFromInts(quant->lstm_i2f_zp), quant->lstm_i2f_qparam);
    inputs_array_data[kLstmInputToForgetWeightsTensor + 1] =
        kLstmInputToForgetWeightsTensor;
  } else if (weight_type == kTfLiteInt8) {
    tensors[kLstmInputToForgetWeightsTensor] =
        CreateQuantizedWeightTensor<int8_t>(
            config->input_to_forget_weights, quant->lstm_i2f_quant,
            IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
            FloatArrayFromFloats(quant->lstm_i2f_scale),
            IntArrayFromInts(quant->lstm_i2f_zp), quant->lstm_i2f_qparam);
    inputs_array_data[kLstmInputToForgetWeightsTensor + 1] =
        kLstmInputToForgetWeightsTensor;
  } else {
    inputs_array_data[kLstmInputToForgetWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  if (weight_type == kTfLiteUInt8) {
    tensors[kLstmInputToCellWeightsTensor] =
        CreateQuantizedWeightTensor<uint8_t>(
            config->input_to_cell_weights,
            reinterpret_cast<uint8_t*>(quant->lstm_i2c_quant),
            IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
            FloatArrayFromFloats(quant->lstm_i2c_scale),
            IntArrayFromInts(quant->lstm_i2c_zp), quant->lstm_i2c_qparam);
    inputs_array_data[kLstmInputToCellWeightsTensor + 1] =
        kLstmInputToCellWeightsTensor;
  } else if (weight_type == kTfLiteInt8) {
    tensors[kLstmInputToCellWeightsTensor] =
        CreateQuantizedWeightTensor<int8_t>(
            config->input_to_cell_weights, quant->lstm_i2c_quant,
            IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
            FloatArrayFromFloats(quant->lstm_i2c_scale),
            IntArrayFromInts(quant->lstm_i2c_zp), quant->lstm_i2c_qparam);
    inputs_array_data[kLstmInputToCellWeightsTensor + 1] =
        kLstmInputToCellWeightsTensor;
  } else {
    inputs_array_data[kLstmInputToCellWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  if (weight_type == kTfLiteUInt8) {
    tensors[kLstmInputToOutputWeightsTensor] =
        CreateQuantizedWeightTensor<uint8_t>(
            config->input_to_output_weights,
            reinterpret_cast<uint8_t*>(quant->lstm_i2o_quant),
            IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
            FloatArrayFromFloats(quant->lstm_i2o_scale),
            IntArrayFromInts(quant->lstm_i2o_zp), quant->lstm_i2o_qparam);
    inputs_array_data[kLstmInputToOutputWeightsTensor + 1] =
        kLstmInputToOutputWeightsTensor;
  } else if (weight_type == kTfLiteInt8) {
    tensors[kLstmInputToOutputWeightsTensor] =
        CreateQuantizedWeightTensor<int8_t>(
            config->input_to_output_weights, quant->lstm_i2o_quant,
            IntArrayFromInts(input_w_dim), config->n_cell * config->n_input,
            FloatArrayFromFloats(quant->lstm_i2o_scale),
            IntArrayFromInts(quant->lstm_i2o_zp), quant->lstm_i2o_qparam);
    inputs_array_data[kLstmInputToOutputWeightsTensor + 1] =
        kLstmInputToOutputWeightsTensor;
  } else {
    inputs_array_data[kLstmInputToOutputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  int recurrent_w_dim[3] = {2, config->n_cell, config->n_output};
  if (config->use_cifg) {
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    if (weight_type == kTfLiteUInt8) {
      tensors[kLstmRecurrentToInputWeightsTensor] =
          CreateQuantizedWeightTensor<uint8_t>(
              config->recurrent_to_input_weights,
              reinterpret_cast<uint8_t*>(quant->lstm_r2i_quant),
              IntArrayFromInts(recurrent_w_dim),
              config->n_cell * config->n_output,
              FloatArrayFromFloats(quant->lstm_r2i_scale),
              IntArrayFromInts(quant->lstm_r2i_zp), quant->lstm_r2i_qparam);
      inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
          kLstmRecurrentToInputWeightsTensor;
    } else if (weight_type == kTfLiteInt8) {
      tensors[kLstmRecurrentToInputWeightsTensor] =
          CreateQuantizedWeightTensor<int8_t>(
              config->recurrent_to_input_weights, quant->lstm_r2i_quant,
              IntArrayFromInts(recurrent_w_dim),
              config->n_cell * config->n_output,
              FloatArrayFromFloats(quant->lstm_r2i_scale),
              IntArrayFromInts(quant->lstm_r2i_zp), quant->lstm_r2i_qparam);
      inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
          kLstmRecurrentToInputWeightsTensor;
    } else {
      inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    }
  }

  if (weight_type == kTfLiteUInt8) {
    tensors[kLstmRecurrentToForgetWeightsTensor] =
        CreateQuantizedWeightTensor<uint8_t>(
            config->recurrent_to_forget_weights,
            reinterpret_cast<uint8_t*>(quant->lstm_r2f_quant),
            IntArrayFromInts(recurrent_w_dim),
            config->n_cell * config->n_output,
            FloatArrayFromFloats(quant->lstm_r2f_scale),
            IntArrayFromInts(quant->lstm_r2f_zp), quant->lstm_r2f_qparam);
    inputs_array_data[kLstmRecurrentToForgetWeightsTensor + 1] =
        kLstmRecurrentToForgetWeightsTensor;
  } else if (weight_type == kTfLiteInt8) {
    tensors[kLstmRecurrentToForgetWeightsTensor] =
        CreateQuantizedWeightTensor<int8_t>(
            config->recurrent_to_forget_weights, quant->lstm_r2f_quant,
            IntArrayFromInts(recurrent_w_dim),
            config->n_cell * config->n_output,
            FloatArrayFromFloats(quant->lstm_r2f_scale),
            IntArrayFromInts(quant->lstm_r2f_zp), quant->lstm_r2f_qparam);
    inputs_array_data[kLstmRecurrentToForgetWeightsTensor + 1] =
        kLstmRecurrentToForgetWeightsTensor;
  } else {
    inputs_array_data[kLstmRecurrentToForgetWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  if (weight_type == kTfLiteUInt8) {
    tensors[kLstmRecurrentToCellWeightsTensor] =
        CreateQuantizedWeightTensor<uint8_t>(
            config->recurrent_to_cell_weights,
            reinterpret_cast<uint8_t*>(quant->lstm_r2c_quant),
            IntArrayFromInts(recurrent_w_dim),
            config->n_cell * config->n_output,
            FloatArrayFromFloats(quant->lstm_r2c_scale),
            IntArrayFromInts(quant->lstm_r2c_zp), quant->lstm_r2c_qparam);
    inputs_array_data[kLstmRecurrentToCellWeightsTensor + 1] =
        kLstmRecurrentToCellWeightsTensor;
  } else if (weight_type == kTfLiteInt8) {
    tensors[kLstmRecurrentToCellWeightsTensor] =
        CreateQuantizedWeightTensor<int8_t>(
            config->recurrent_to_cell_weights, quant->lstm_r2c_quant,
            IntArrayFromInts(recurrent_w_dim),
            config->n_cell * config->n_output,
            FloatArrayFromFloats(quant->lstm_r2c_scale),
            IntArrayFromInts(quant->lstm_r2c_zp), quant->lstm_r2c_qparam);
    inputs_array_data[kLstmRecurrentToCellWeightsTensor + 1] =
        kLstmRecurrentToCellWeightsTensor;
  } else {
    inputs_array_data[kLstmRecurrentToCellWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  if (weight_type == kTfLiteUInt8) {
    tensors[kLstmRecurrentToOutputWeightsTensor] =
        CreateQuantizedWeightTensor<uint8_t>(
            config->recurrent_to_output_weights,
            reinterpret_cast<uint8_t*>(quant->lstm_r2o_quant),
            IntArrayFromInts(recurrent_w_dim),
            config->n_cell * config->n_output,
            FloatArrayFromFloats(quant->lstm_r2o_scale),
            IntArrayFromInts(quant->lstm_r2o_zp), quant->lstm_r2o_qparam);
    inputs_array_data[kLstmRecurrentToOutputWeightsTensor + 1] =
        kLstmRecurrentToOutputWeightsTensor;
  } else if (weight_type == kTfLiteInt8) {
    tensors[kLstmRecurrentToOutputWeightsTensor] =
        CreateQuantizedWeightTensor<int8_t>(
            config->recurrent_to_output_weights, quant->lstm_r2o_quant,
            IntArrayFromInts(recurrent_w_dim),
            config->n_cell * config->n_output,
            FloatArrayFromFloats(quant->lstm_r2o_scale),
            IntArrayFromInts(quant->lstm_r2o_zp), quant->lstm_r2o_qparam);
    inputs_array_data[kLstmRecurrentToOutputWeightsTensor + 1] =
        kLstmRecurrentToOutputWeightsTensor;
  } else {
    inputs_array_data[kLstmRecurrentToOutputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  int cell_w_dim[2] = {1, config->n_cell};
  if (config->use_peephole) {
    if (config->use_cifg) {
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      if (weight_type == kTfLiteUInt8) {
        tensors[kLstmCellToInputWeightsTensor] =
            CreateQuantizedWeightTensor<uint8_t>(
                config->cell_to_input_weights,
                reinterpret_cast<uint8_t*>(quant->lstm_c2i_quant),
                IntArrayFromInts(cell_w_dim), config->n_cell,
                FloatArrayFromFloats(quant->lstm_c2i_scale),
                IntArrayFromInts(quant->lstm_c2i_zp), quant->lstm_c2i_qparam);
        inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
            kLstmCellToInputWeightsTensor;
      } else if (weight_type == kTfLiteInt8) {
        tensors[kLstmCellToInputWeightsTensor] =
            CreateQuantizedWeightTensor<int8_t>(
                config->cell_to_input_weights, quant->lstm_c2i_quant,
                IntArrayFromInts(cell_w_dim), config->n_cell,
                FloatArrayFromFloats(quant->lstm_c2i_scale),
                IntArrayFromInts(quant->lstm_c2i_zp), quant->lstm_c2i_qparam);
        inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
            kLstmCellToInputWeightsTensor;
      } else {
        inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
            kTfLiteOptionalTensor;
      }
    }
    if (weight_type == kTfLiteUInt8) {
      tensors[kLstmCellToForgetWeightsTensor] =
          CreateQuantizedWeightTensor<uint8_t>(
              config->cell_to_forget_weights,
              reinterpret_cast<uint8_t*>(quant->lstm_c2f_quant),
              IntArrayFromInts(cell_w_dim), config->n_cell,
              FloatArrayFromFloats(quant->lstm_c2f_scale),
              IntArrayFromInts(quant->lstm_c2f_zp), quant->lstm_c2f_qparam);
      inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
          kLstmCellToForgetWeightsTensor;
    } else if (weight_type == kTfLiteInt8) {
      tensors[kLstmCellToForgetWeightsTensor] =
          CreateQuantizedWeightTensor<int8_t>(
              config->cell_to_forget_weights, quant->lstm_c2f_quant,
              IntArrayFromInts(cell_w_dim), config->n_cell,
              FloatArrayFromFloats(quant->lstm_c2f_scale),
              IntArrayFromInts(quant->lstm_c2f_zp), quant->lstm_c2f_qparam);
      inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
          kLstmCellToForgetWeightsTensor;
    } else {
      inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    }

    if (weight_type == kTfLiteUInt8) {
      tensors[kLstmCellToOutputWeightsTensor] =
          CreateQuantizedWeightTensor<uint8_t>(
              config->cell_to_output_weights,
              reinterpret_cast<uint8_t*>(quant->lstm_c2o_quant),
              IntArrayFromInts(cell_w_dim), config->n_cell,
              FloatArrayFromFloats(quant->lstm_c2o_scale),
              IntArrayFromInts(quant->lstm_c2o_zp), quant->lstm_c2o_qparam);
      inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
          kLstmCellToOutputWeightsTensor;
    } else if (weight_type == kTfLiteInt8) {
      tensors[kLstmCellToOutputWeightsTensor] =
          CreateQuantizedWeightTensor<int8_t>(
              config->cell_to_output_weights, quant->lstm_c2o_quant,
              IntArrayFromInts(cell_w_dim), config->n_cell,
              FloatArrayFromFloats(quant->lstm_c2o_scale),
              IntArrayFromInts(quant->lstm_c2o_zp), quant->lstm_c2o_qparam);
      inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
          kLstmCellToOutputWeightsTensor;
    } else {
      inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    }
  } else {
    inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  int gate_bias_dim[2] = {1, config->n_cell};
  if (config->use_cifg) {
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kTfLiteOptionalTensor;
  } else {
    tensors[kLstmInputGateBiasTensor] = CreateTensor<float>(
        config->input_gate_bias, IntArrayFromInts(gate_bias_dim));
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kLstmInputGateBiasTensor;
  }

  tensors[kLstmForgetGateBiasTensor] = CreateTensor<float>(
      config->forget_gate_bias, IntArrayFromInts(gate_bias_dim));
  inputs_array_data[kLstmForgetGateBiasTensor + 1] = kLstmForgetGateBiasTensor;

  tensors[kLstmCellGateBiasTensor] = CreateTensor<float>(
      config->cell_gate_bias, IntArrayFromInts(gate_bias_dim));
  inputs_array_data[kLstmCellGateBiasTensor + 1] = kLstmCellGateBiasTensor;

  tensors[kLstmOutputGateBiasTensor] = CreateTensor<float>(
      config->output_gate_bias, IntArrayFromInts(gate_bias_dim));
  inputs_array_data[kLstmOutputGateBiasTensor + 1] = kLstmOutputGateBiasTensor;

  int lstm_proj_w_dim[3] = {2, config->n_output, config->n_cell};
  int projection_bias_dim[2] = {1, config->n_output};
  if (config->use_projection_weights) {
    if (weight_type == kTfLiteUInt8) {
      tensors[kLstmProjectionWeightsTensor] =
          CreateQuantizedWeightTensor<uint8_t>(
              config->projection_weights,
              reinterpret_cast<uint8_t*>(quant->lstm_proj_w_quant),
              IntArrayFromInts(lstm_proj_w_dim),
              config->n_output * config->n_cell,
              FloatArrayFromFloats(quant->lstm_proj_w_scale),
              IntArrayFromInts(quant->lstm_proj_w_zp),
              quant->lstm_proj_w_qparam);
      inputs_array_data[kLstmProjectionWeightsTensor + 1] =
          kLstmProjectionWeightsTensor;
    } else if (weight_type == kTfLiteInt8) {
      tensors[kLstmProjectionWeightsTensor] =
          CreateQuantizedWeightTensor<int8_t>(
              config->projection_weights, quant->lstm_proj_w_quant,
              IntArrayFromInts(lstm_proj_w_dim),
              config->n_output * config->n_cell,
              FloatArrayFromFloats(quant->lstm_proj_w_scale),
              IntArrayFromInts(quant->lstm_proj_w_zp),
              quant->lstm_proj_w_qparam);
      inputs_array_data[kLstmProjectionWeightsTensor + 1] =
          kLstmProjectionWeightsTensor;
    } else {
      inputs_array_data[kLstmProjectionWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    }

    if (config->use_projection_bias) {
      tensors[kLstmProjectionBiasTensor] = CreateTensor<float>(
          config->projection_bias, IntArrayFromInts(projection_bias_dim));
      inputs_array_data[kLstmProjectionBiasTensor + 1] =
          kLstmProjectionBiasTensor;
    } else {
      inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
    }
  } else {
    inputs_array_data[kLstmProjectionWeightsTensor + 1] = kTfLiteOptionalTensor;
    inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
  }

  int lstm_output_statedim[3] = {2, config->n_batch, config->n_output};
  for (int i = 0; i < config->n_batch; ++i) {
    for (int j = 0; j < config->n_output; ++j) {
      config->output_state[i * config->n_output + j] = 0.0f;
    }
  }
  tensors[kLstmOutputStateTensor] = CreateTensor<float>(
      config->output_state, IntArrayFromInts(lstm_output_statedim), true);
  inputs_array_data[kLstmOutputStateTensor + 1] = kLstmOutputStateTensor;

  int lstm_cell_state_dim[3] = {2, config->n_batch, config->n_cell};
  for (int i = 0; i < config->n_batch; ++i) {
    for (int j = 0; j < config->n_cell; ++j) {
      config->cell_state[i * config->n_cell + j] = 0.0f;
    }
  }
  tensors[kLstmCellStateTensor] = CreateTensor<float>(
      config->cell_state, IntArrayFromInts(lstm_cell_state_dim), true);
  inputs_array_data[kLstmCellStateTensor + 1] = kLstmCellStateTensor;

  int layer_norm_dim[2] = {1, config->n_cell};
  if (config->use_layer_norm) {
    if (config->use_cifg) {
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      tensors[kLstmInputLayerNormCoefficientsTensor] =
          CreateTensor<float>(config->input_layer_norm_coefficients,
                              IntArrayFromInts(layer_norm_dim));
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kLstmInputLayerNormCoefficientsTensor;
    }

    tensors[kLstmForgetLayerNormCoefficientsTensor] =
        CreateTensor<float>(config->forget_layer_norm_coefficients,
                            IntArrayFromInts(layer_norm_dim));
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kLstmForgetLayerNormCoefficientsTensor;

    tensors[kLstmCellLayerNormCoefficientsTensor] = CreateTensor<float>(
        config->cell_layer_norm_coefficients, IntArrayFromInts(layer_norm_dim));
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kLstmCellLayerNormCoefficientsTensor;

    tensors[kLstmOutputLayerNormCoefficientsTensor] =
        CreateTensor<float>(config->output_layer_norm_coefficients,
                            IntArrayFromInts(layer_norm_dim));
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kLstmOutputLayerNormCoefficientsTensor;
  } else {
    inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  CopyInputOrExpectedOutput(config->expected_output_original,
                            config->expected_output, input_output_batch_major,
                            config->sequence_length, config->n_batch,
                            config->n_output);

  int output_dim[4] = {3, config->sequence_length, config->n_batch,
                       config->n_output};
  for (int i = 0; i < config->sequence_length; ++i) {
    for (int j = 0; j < config->n_batch; ++j) {
      for (int k = 0; k < config->n_output; ++k) {
        config->output[i * config->n_batch * config->n_output +
                       j * config->n_output + k] = 0.0f;
      }
    }
  }
  tensors[kLstmOutputTensorIndex] =
      CreateTensor<float>(config->output, IntArrayFromInts(output_dim));

  TfLiteUnidirectionalSequenceLSTMParams params;
  params.activation = kTfLiteActTanh;
  params.cell_clip = config->cell_clip;
  params.proj_clip = config->proj_clip;
  params.time_major = config->time_major;
  params.asymmetric_quantize_inputs = asymmetric_quantize_inputs;

  const TfLiteRegistration registration =
      Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  micro::KernelRunner runner(registration, tensors, kLstmMaxNumInputTensors + 1,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data),
                             reinterpret_cast<void*>(&params));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0;
       i < config->sequence_length * config->n_batch * config->n_output; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(config->expected_output[i], config->output[i],
                              tolerance);
  }
}

void TestUnidirectionalSequenceLstmFloat(
    LstmFloatTestConfig* config, float tolerance, bool input_output_batch_major,
    bool asymmetric_quantize_inputs = false) {
  int inputs_array_data[25];
  int outputs_array_data[2] = {1, kLstmOutputTensorIndex};

  if (config->use_layer_norm) {
    inputs_array_data[0] = 24;
  } else {
    inputs_array_data[0] = 20;
  }

  TfLiteTensor tensors[kLstmMaxNumInputTensors + 1];

  CopyInputOrExpectedOutput(config->input_original, config->input,
                            input_output_batch_major, config->sequence_length,
                            config->n_batch, config->n_input);

  int input_dim[4] = {3, config->sequence_length, config->n_batch,
                      config->n_input};
  tensors[kLstmInputTensor] =
      CreateTensor<float>(config->input, IntArrayFromInts(input_dim));
  inputs_array_data[kLstmInputTensor + 1] = kLstmInputTensor;

  int input_w_dim[3] = {2, config->n_cell, config->n_input};

  if (config->use_cifg) {
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    tensors[kLstmInputToInputWeightsTensor] = CreateTensor<float>(
        config->input_to_input_weights, IntArrayFromInts(input_w_dim));
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kLstmInputToInputWeightsTensor;
  }

  tensors[kLstmInputToForgetWeightsTensor] = CreateTensor<float>(
      config->input_to_forget_weights, IntArrayFromInts(input_w_dim));
  inputs_array_data[kLstmInputToForgetWeightsTensor + 1] =
      kLstmInputToForgetWeightsTensor;

  tensors[kLstmInputToCellWeightsTensor] = CreateTensor<float>(
      config->input_to_cell_weights, IntArrayFromInts(input_w_dim));
  inputs_array_data[kLstmInputToCellWeightsTensor + 1] =
      kLstmInputToCellWeightsTensor;

  tensors[kLstmInputToOutputWeightsTensor] = CreateTensor<float>(
      config->input_to_output_weights, IntArrayFromInts(input_w_dim));
  inputs_array_data[kLstmInputToOutputWeightsTensor + 1] =
      kLstmInputToOutputWeightsTensor;

  int recurrent_w_dim[3] = {2, config->n_cell, config->n_output};
  if (config->use_cifg) {
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    tensors[kLstmRecurrentToInputWeightsTensor] = CreateTensor<float>(
        config->recurrent_to_input_weights, IntArrayFromInts(recurrent_w_dim));
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kLstmRecurrentToInputWeightsTensor;
  }

  tensors[kLstmRecurrentToForgetWeightsTensor] = CreateTensor<float>(
      config->recurrent_to_forget_weights, IntArrayFromInts(recurrent_w_dim));
  inputs_array_data[kLstmRecurrentToForgetWeightsTensor + 1] =
      kLstmRecurrentToForgetWeightsTensor;

  tensors[kLstmRecurrentToCellWeightsTensor] = CreateTensor<float>(
      config->recurrent_to_cell_weights, IntArrayFromInts(recurrent_w_dim));
  inputs_array_data[kLstmRecurrentToCellWeightsTensor + 1] =
      kLstmRecurrentToCellWeightsTensor;

  tensors[kLstmRecurrentToOutputWeightsTensor] = CreateTensor<float>(
      config->recurrent_to_output_weights, IntArrayFromInts(recurrent_w_dim));
  inputs_array_data[kLstmRecurrentToOutputWeightsTensor + 1] =
      kLstmRecurrentToOutputWeightsTensor;

  int cell_w_dim[2] = {1, config->n_cell};
  if (config->use_peephole) {
    if (config->use_cifg) {
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      tensors[kLstmCellToInputWeightsTensor] = CreateTensor<float>(
          config->cell_to_input_weights, IntArrayFromInts(cell_w_dim));
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kLstmCellToInputWeightsTensor;
    }

    tensors[kLstmCellToForgetWeightsTensor] = CreateTensor<float>(
        config->cell_to_forget_weights, IntArrayFromInts(cell_w_dim));
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kLstmCellToForgetWeightsTensor;

    tensors[kLstmCellToOutputWeightsTensor] = CreateTensor<float>(
        config->cell_to_output_weights, IntArrayFromInts(cell_w_dim));
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kLstmCellToOutputWeightsTensor;
  } else {
    inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  int gate_bias_dim[2] = {1, config->n_cell};
  if (config->use_cifg) {
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kTfLiteOptionalTensor;
  } else {
    tensors[kLstmInputGateBiasTensor] = CreateTensor<float>(
        config->input_gate_bias, IntArrayFromInts(gate_bias_dim));
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kLstmInputGateBiasTensor;
  }

  tensors[kLstmForgetGateBiasTensor] = CreateTensor<float>(
      config->forget_gate_bias, IntArrayFromInts(gate_bias_dim));
  inputs_array_data[kLstmForgetGateBiasTensor + 1] = kLstmForgetGateBiasTensor;

  tensors[kLstmCellGateBiasTensor] = CreateTensor<float>(
      config->cell_gate_bias, IntArrayFromInts(gate_bias_dim));
  inputs_array_data[kLstmCellGateBiasTensor + 1] = kLstmCellGateBiasTensor;

  tensors[kLstmOutputGateBiasTensor] = CreateTensor<float>(
      config->output_gate_bias, IntArrayFromInts(gate_bias_dim));
  inputs_array_data[kLstmOutputGateBiasTensor + 1] = kLstmOutputGateBiasTensor;

  int lstm_proj_w_dim[3] = {2, config->n_output, config->n_cell};
  int projection_bias_dim[2] = {1, config->n_output};
  if (config->use_projection_weights) {
    tensors[kLstmProjectionWeightsTensor] = CreateTensor<float>(
        config->projection_weights, IntArrayFromInts(lstm_proj_w_dim));
    inputs_array_data[kLstmProjectionWeightsTensor + 1] =
        kLstmProjectionWeightsTensor;

    if (config->use_projection_bias) {
      tensors[kLstmProjectionBiasTensor] = CreateTensor<float>(
          config->projection_bias, IntArrayFromInts(projection_bias_dim));
      inputs_array_data[kLstmProjectionBiasTensor + 1] =
          kLstmProjectionBiasTensor;
    } else {
      inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
    }
  } else {
    inputs_array_data[kLstmProjectionWeightsTensor + 1] = kTfLiteOptionalTensor;
    inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
  }

  int lstm_output_statedim[3] = {2, config->n_batch, config->n_output};
  for (int i = 0; i < config->n_batch; ++i) {
    for (int j = 0; j < config->n_output; ++j) {
      config->output_state[i * config->n_output + j] = 0.0f;
    }
  }
  tensors[kLstmOutputStateTensor] = CreateTensor<float>(
      config->output_state, IntArrayFromInts(lstm_output_statedim), true);
  inputs_array_data[kLstmOutputStateTensor + 1] = kLstmOutputStateTensor;

  int lstm_cell_state_dim[3] = {2, config->n_batch, config->n_cell};
  for (int i = 0; i < config->n_batch; ++i) {
    for (int j = 0; j < config->n_cell; ++j) {
      config->cell_state[i * config->n_cell + j] = 0.0f;
    }
  }
  tensors[kLstmCellStateTensor] = CreateTensor<float>(
      config->cell_state, IntArrayFromInts(lstm_cell_state_dim), true);
  inputs_array_data[kLstmCellStateTensor + 1] = kLstmCellStateTensor;

  int layer_norm_dim[2] = {1, config->n_cell};
  if (config->use_layer_norm) {
    if (config->use_cifg) {
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      tensors[kLstmInputLayerNormCoefficientsTensor] =
          CreateTensor<float>(config->input_layer_norm_coefficients,
                              IntArrayFromInts(layer_norm_dim));
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kLstmInputLayerNormCoefficientsTensor;
    }

    tensors[kLstmForgetLayerNormCoefficientsTensor] =
        CreateTensor<float>(config->forget_layer_norm_coefficients,
                            IntArrayFromInts(layer_norm_dim));
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kLstmForgetLayerNormCoefficientsTensor;

    tensors[kLstmCellLayerNormCoefficientsTensor] = CreateTensor<float>(
        config->cell_layer_norm_coefficients, IntArrayFromInts(layer_norm_dim));
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kLstmCellLayerNormCoefficientsTensor;

    tensors[kLstmOutputLayerNormCoefficientsTensor] =
        CreateTensor<float>(config->output_layer_norm_coefficients,
                            IntArrayFromInts(layer_norm_dim));
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kLstmOutputLayerNormCoefficientsTensor;
  } else {
    inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  CopyInputOrExpectedOutput(config->expected_output_original,
                            config->expected_output, input_output_batch_major,
                            config->sequence_length, config->n_batch,
                            config->n_output);

  int output_dim[4] = {3, config->sequence_length, config->n_batch,
                       config->n_output};
  for (int i = 0; i < config->sequence_length; ++i) {
    for (int j = 0; j < config->n_batch; ++j) {
      for (int k = 0; k < config->n_output; ++k) {
        config->output[i * config->n_batch * config->n_output +
                       j * config->n_output + k] = 0.0f;
      }
    }
  }
  tensors[kLstmOutputTensorIndex] =
      CreateTensor<float>(config->output, IntArrayFromInts(output_dim));

  TfLiteUnidirectionalSequenceLSTMParams params;
  params.activation = kTfLiteActTanh;
  params.cell_clip = config->cell_clip;
  params.proj_clip = config->proj_clip;
  params.time_major = config->time_major;
  params.asymmetric_quantize_inputs = asymmetric_quantize_inputs;

  const TfLiteRegistration registration =
      Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  micro::KernelRunner runner(registration, tensors, kLstmMaxNumInputTensors + 1,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data),
                             reinterpret_cast<void*>(&params));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0;
       i < config->sequence_length * config->n_batch * config->n_output; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(config->expected_output[i], config->output[i],
                              tolerance);
  }
}

void TestUnidirectionalSequenceLstmInteger(LstmIntegerTestConfig* config) {
  int inputs_array_data[25];
  int outputs_array_data[2] = {1, kLstmOutputTensorIndex};
  int intermediate_array_data[6] = {5,
                                    kLstmIntermediateTensorBase,
                                    kLstmIntermediateTensorBase + 1,
                                    kLstmIntermediateTensorBase + 2,
                                    kLstmIntermediateTensorBase + 3,
                                    kLstmIntermediateTensorBase + 4};

  if (config->use_layer_norm) {
    inputs_array_data[0] = 24;
  } else {
    inputs_array_data[0] = 20;
  }

  QuantizationParams quantization_params;

  TfLiteTensor tensors[kLstmMaxNumInputTensors + 1 + 5];

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmInputTensor][0], config->ranges[kLstmInputTensor][1]);
  int input_dim[4] = {3, config->sequence_length, config->n_batch,
                      config->n_input};
  tensors[kLstmInputTensor] = CreateQuantizedTensor<int8_t>(
      config->input, config->input_quant, IntArrayFromInts(input_dim),
      quantization_params.scale, quantization_params.zero_point);
  inputs_array_data[kLstmInputTensor + 1] = kLstmInputTensor;

  int input_w_dim[3] = {2, config->n_cell, config->n_input};

  if (config->use_cifg) {
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    quantization_params = SetQuantizationParams<int8_t>(
        config->ranges[kLstmInputToInputWeightsTensor][0],
        config->ranges[kLstmInputToInputWeightsTensor][1]);
    tensors[kLstmInputToInputWeightsTensor] = CreateQuantizedTensor<int8_t>(
        config->input_to_input_weights, config->lstm_i2i_quant,
        IntArrayFromInts(input_w_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kLstmInputToInputWeightsTensor;
  }

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmInputToForgetWeightsTensor][0],
      config->ranges[kLstmInputToForgetWeightsTensor][1]);
  tensors[kLstmInputToForgetWeightsTensor] = CreateQuantizedTensor<int8_t>(
      config->input_to_forget_weights, config->lstm_i2f_quant,
      IntArrayFromInts(input_w_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmInputToForgetWeightsTensor + 1] =
      kLstmInputToForgetWeightsTensor;

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmInputToCellWeightsTensor][0],
      config->ranges[kLstmInputToCellWeightsTensor][1]);
  tensors[kLstmInputToCellWeightsTensor] = CreateQuantizedTensor<int8_t>(
      config->input_to_cell_weights, config->lstm_i2c_quant,
      IntArrayFromInts(input_w_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmInputToCellWeightsTensor + 1] =
      kLstmInputToCellWeightsTensor;

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmInputToOutputWeightsTensor][0],
      config->ranges[kLstmInputToOutputWeightsTensor][1]);
  tensors[kLstmInputToOutputWeightsTensor] = CreateQuantizedTensor<int8_t>(
      config->input_to_output_weights, config->lstm_i2o_quant,
      IntArrayFromInts(input_w_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmInputToOutputWeightsTensor + 1] =
      kLstmInputToOutputWeightsTensor;

  int recurrent_w_dim[3] = {2, config->n_cell, config->n_output};
  if (config->use_cifg) {
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    quantization_params = SetQuantizationParams<int8_t>(
        config->ranges[kLstmRecurrentToInputWeightsTensor][0],
        config->ranges[kLstmRecurrentToInputWeightsTensor][1]);
    tensors[kLstmRecurrentToInputWeightsTensor] = CreateQuantizedTensor<int8_t>(
        config->recurrent_to_input_weights, config->lstm_r2i_quant,
        IntArrayFromInts(recurrent_w_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kLstmRecurrentToInputWeightsTensor;
  }

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmRecurrentToForgetWeightsTensor][0],
      config->ranges[kLstmRecurrentToForgetWeightsTensor][1]);
  tensors[kLstmRecurrentToForgetWeightsTensor] = CreateQuantizedTensor<int8_t>(
      config->recurrent_to_forget_weights, config->lstm_r2f_quant,
      IntArrayFromInts(recurrent_w_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmRecurrentToForgetWeightsTensor + 1] =
      kLstmRecurrentToForgetWeightsTensor;

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmRecurrentToCellWeightsTensor][0],
      config->ranges[kLstmRecurrentToCellWeightsTensor][1]);
  tensors[kLstmRecurrentToCellWeightsTensor] = CreateQuantizedTensor<int8_t>(
      config->recurrent_to_cell_weights, config->lstm_r2c_quant,
      IntArrayFromInts(recurrent_w_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmRecurrentToCellWeightsTensor + 1] =
      kLstmRecurrentToCellWeightsTensor;

  quantization_params = SetQuantizationParams<int8_t>(
      config->ranges[kLstmRecurrentToOutputWeightsTensor][0],
      config->ranges[kLstmRecurrentToOutputWeightsTensor][1]);
  tensors[kLstmRecurrentToOutputWeightsTensor] = CreateQuantizedTensor<int8_t>(
      config->recurrent_to_output_weights, config->lstm_r2o_quant,
      IntArrayFromInts(recurrent_w_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmRecurrentToOutputWeightsTensor + 1] =
      kLstmRecurrentToOutputWeightsTensor;

  int cell_w_dim[2] = {1, config->n_cell};
  if (config->use_peephole) {
    if (config->use_cifg) {
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      quantization_params = SetQuantizationParams<int16_t>(
          config->ranges[kLstmCellToInputWeightsTensor][0],
          config->ranges[kLstmCellToInputWeightsTensor][1]);
      tensors[kLstmCellToInputWeightsTensor] = CreateQuantizedTensor<int16_t>(
          config->cell_to_input_weights, config->lstm_c2i_quant,
          IntArrayFromInts(cell_w_dim), quantization_params.scale,
          quantization_params.zero_point);
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kLstmCellToInputWeightsTensor;
    }

    quantization_params = SetQuantizationParams<int16_t>(
        config->ranges[kLstmCellToForgetWeightsTensor][0],
        config->ranges[kLstmCellToForgetWeightsTensor][1]);
    tensors[kLstmCellToForgetWeightsTensor] = CreateQuantizedTensor<int16_t>(
        config->cell_to_forget_weights, config->lstm_c2f_quant,
        IntArrayFromInts(cell_w_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kLstmCellToForgetWeightsTensor;

    quantization_params = SetQuantizationParams<int16_t>(
        config->ranges[kLstmCellToOutputWeightsTensor][0],
        config->ranges[kLstmCellToOutputWeightsTensor][1]);
    tensors[kLstmCellToOutputWeightsTensor] = CreateQuantizedTensor<int16_t>(
        config->cell_to_output_weights, config->lstm_c2o_quant,
        IntArrayFromInts(cell_w_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kLstmCellToOutputWeightsTensor;
  } else {
    inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  int gate_bias_dim[2] = {1, config->n_cell};
  if (config->use_cifg) {
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kTfLiteOptionalTensor;
  } else {
    quantization_params = SetQuantizationParams<int32_t>(
        config->ranges[kLstmInputGateBiasTensor][0],
        config->ranges[kLstmInputGateBiasTensor][1]);
    tensors[kLstmInputGateBiasTensor] = CreateQuantizedTensor<int32_t>(
        config->input_gate_bias, config->lstm_igate_bias_quant,
        IntArrayFromInts(gate_bias_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kLstmInputGateBiasTensor;
  }

  quantization_params = SetQuantizationParams<int32_t>(
      config->ranges[kLstmForgetGateBiasTensor][0],
      config->ranges[kLstmForgetGateBiasTensor][1]);
  tensors[kLstmForgetGateBiasTensor] = CreateQuantizedTensor<int32_t>(
      config->forget_gate_bias, config->lstm_fgate_bias_quant,
      IntArrayFromInts(gate_bias_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmForgetGateBiasTensor + 1] = kLstmForgetGateBiasTensor;

  quantization_params = SetQuantizationParams<int32_t>(
      config->ranges[kLstmCellGateBiasTensor][0],
      config->ranges[kLstmCellGateBiasTensor][1]);
  tensors[kLstmCellGateBiasTensor] = CreateQuantizedTensor<int32_t>(
      config->cell_gate_bias, config->lstm_cgate_bias_quant,
      IntArrayFromInts(gate_bias_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmCellGateBiasTensor + 1] = kLstmCellGateBiasTensor;

  quantization_params = SetQuantizationParams<int32_t>(
      config->ranges[kLstmOutputGateBiasTensor][0],
      config->ranges[kLstmOutputGateBiasTensor][1]);
  tensors[kLstmOutputGateBiasTensor] = CreateQuantizedTensor<int32_t>(
      config->output_gate_bias, config->lstm_ogate_bias_quant,
      IntArrayFromInts(gate_bias_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmOutputGateBiasTensor + 1] = kLstmOutputGateBiasTensor;

  int lstm_proj_w_dim[3] = {2, config->n_output, config->n_cell};
  if (config->use_projection_weights) {
    quantization_params = SetQuantizationParams<int8_t>(
        config->ranges[kLstmProjectionWeightsTensor][0],
        config->ranges[kLstmProjectionWeightsTensor][1]);
    tensors[kLstmProjectionWeightsTensor] = CreateQuantizedTensor<int8_t>(
        config->projection_weights, config->lstm_proj_w_quant,
        IntArrayFromInts(lstm_proj_w_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmProjectionWeightsTensor + 1] =
        kLstmProjectionWeightsTensor;

    int projection_bias_dim[2] = {1, config->n_output};
    if (config->use_projection_bias) {
      quantization_params = SetQuantizationParams<int32_t>(
          config->ranges[kLstmProjectionBiasTensor][0],
          config->ranges[kLstmProjectionBiasTensor][1]);
      tensors[kLstmProjectionBiasTensor] = CreateQuantizedTensor<int32_t>(
          config->projection_bias, config->projection_bias_quant,
          IntArrayFromInts(projection_bias_dim), quantization_params.scale,
          quantization_params.zero_point);
      inputs_array_data[kLstmProjectionBiasTensor + 1] =
          kLstmProjectionBiasTensor;
    } else {
      inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
    }
  } else {
    inputs_array_data[kLstmProjectionWeightsTensor + 1] = kTfLiteOptionalTensor;
    inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
  }

  int lstm_output_statedim[3] = {2, config->n_batch, config->n_output};
  quantization_params =
      SetQuantizationParams<int16_t>(config->ranges[kLstmOutputStateTensor][0],
                                     config->ranges[kLstmOutputStateTensor][1]);
  tensors[kLstmOutputStateTensor] = CreateQuantizedTensor<int16_t>(
      config->output_state, IntArrayFromInts(lstm_output_statedim),
      quantization_params.scale, quantization_params.zero_point, true);
  inputs_array_data[kLstmOutputStateTensor + 1] = kLstmOutputStateTensor;

  int lstm_cell_state_dim[3] = {2, config->n_batch, config->n_cell};
  quantization_params =
      SetQuantizationParams<int16_t>(config->ranges[kLstmCellStateTensor][0],
                                     config->ranges[kLstmCellStateTensor][1]);
  tensors[kLstmCellStateTensor] = CreateQuantizedTensor<int16_t>(
      config->cell_state, IntArrayFromInts(lstm_cell_state_dim),
      quantization_params.scale, quantization_params.zero_point, true);
  inputs_array_data[kLstmCellStateTensor + 1] = kLstmCellStateTensor;

  int layer_norm_dim[2] = {1, config->n_cell};
  if (config->use_layer_norm) {
    if (config->use_cifg) {
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      quantization_params = SetQuantizationParams<int16_t>(
          config->ranges[kLstmInputLayerNormCoefficientsTensor][0],
          config->ranges[kLstmInputLayerNormCoefficientsTensor][1]);
      tensors[kLstmInputLayerNormCoefficientsTensor] =
          CreateQuantizedTensor<int16_t>(
              config->input_layer_norm_coefficients,
              config->lstm_input_layer_norm_coeff_quant,
              IntArrayFromInts(layer_norm_dim), quantization_params.scale,
              quantization_params.zero_point);
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kLstmInputLayerNormCoefficientsTensor;
    }

    quantization_params = SetQuantizationParams<int16_t>(
        config->ranges[kLstmForgetLayerNormCoefficientsTensor][0],
        config->ranges[kLstmForgetLayerNormCoefficientsTensor][1]);
    tensors[kLstmForgetLayerNormCoefficientsTensor] =
        CreateQuantizedTensor<int16_t>(
            config->forget_layer_norm_coefficients,
            config->lstm_forget_layer_norm_coeff_quant,
            IntArrayFromInts(layer_norm_dim), quantization_params.scale,
            quantization_params.zero_point);
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kLstmForgetLayerNormCoefficientsTensor;

    quantization_params = SetQuantizationParams<int16_t>(
        config->ranges[kLstmCellLayerNormCoefficientsTensor][0],
        config->ranges[kLstmCellLayerNormCoefficientsTensor][1]);
    tensors[kLstmCellLayerNormCoefficientsTensor] =
        CreateQuantizedTensor<int16_t>(config->cell_layer_norm_coefficients,
                                       config->lstm_cell_layer_norm_coeff_quant,
                                       IntArrayFromInts(layer_norm_dim),
                                       quantization_params.scale,
                                       quantization_params.zero_point);
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kLstmCellLayerNormCoefficientsTensor;

    quantization_params = SetQuantizationParams<int16_t>(
        config->ranges[kLstmOutputLayerNormCoefficientsTensor][0],
        config->ranges[kLstmOutputLayerNormCoefficientsTensor][1]);
    tensors[kLstmOutputLayerNormCoefficientsTensor] =
        CreateQuantizedTensor<int16_t>(
            config->output_layer_norm_coefficients,
            config->lstm_output_layer_norm_coeff_quant,
            IntArrayFromInts(layer_norm_dim), quantization_params.scale,
            quantization_params.zero_point);
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kLstmOutputLayerNormCoefficientsTensor;
  } else {
    inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
  }
  int output_dim[4] = {3, config->sequence_length, config->n_batch,
                       config->n_output};
  quantization_params =
      SetQuantizationParams<int8_t>(config->ranges[kLstmOutputTensorIndex][0],
                                    config->ranges[kLstmOutputTensorIndex][1]);
  tensors[kLstmOutputTensorIndex] = CreateQuantizedTensor<int8_t>(
      config->output, IntArrayFromInts(output_dim), quantization_params.scale,
      quantization_params.zero_point);

  int intermediate_dim[2] = {1, 0};
  for (int i = 0; i < 5; ++i) {
    tensors[kLstmIntermediateTensorBase + i] =
        CreateTensor<int16_t>(nullptr, IntArrayFromInts(intermediate_dim));
    config->intermediate_qparam[i].scale =
        FloatArrayFromFloats(config->intermediate_scale[i]);
    config->intermediate_qparam[i].zero_point =
        IntArrayFromInts(config->intermediate_zp[i]);
    config->intermediate_qparam[i].quantized_dimension = 0;
    tensors[kLstmIntermediateTensorBase + i].quantization.params =
        &config->intermediate_qparam[i];
  }

  TfLiteUnidirectionalSequenceLSTMParams params;
  params.activation = kTfLiteActTanh;
  params.cell_clip = 0.0f;
  params.proj_clip = 0.0f;
  params.time_major = config->time_major;
  params.asymmetric_quantize_inputs = config->asymmetric_quantize_inputs;

  const TfLiteRegistration registration =
      Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  micro::KernelRunner runner(
      registration, tensors, kLstmMaxNumInputTensors + 1 + 5,
      IntArrayFromInts(inputs_array_data), IntArrayFromInts(outputs_array_data),
      reinterpret_cast<void*>(&params),
      IntArrayFromInts(intermediate_array_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0;
       i < config->sequence_length * config->n_batch * config->n_output; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(config->expected_output[i], config->output[i]);
  }
}

#endif  // !defined(XTENSA)
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)

TF_LITE_MICRO_TEST(UnidirectionalSequenceLstmIntegerNoPeepholeTest) {
  tflite::testing::TestUnidirectionalSequenceLstmInteger(
      &tflite::testing::lstm_integer_no_peephole_config);
}

TF_LITE_MICRO_TEST(UnidirectionalSequenceLstmIntegerPeepholeTest) {
  tflite::testing::TestUnidirectionalSequenceLstmInteger(
      &tflite::testing::lstm_integer_peephole_config);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmFloatNoCifgNoPphlNoPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmFloat(
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_config,

      /*tolerance=*/1e-5,
      /*input_output_batch_major=*/false);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridInt8NoCifgNoPphlNoPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_config,
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_buffers,
      /*tolerance=*/0.0157651,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteInt8);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridInt8NoCifgNoPphlNoPrjAsymTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_config,
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_buffers,
      /*tolerance=*/0.0157651,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteInt8,
      /*asymmetric_quantize_inputs=*/true);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridUInt8NoCifgNoPphlNoPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_config,
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_buffers,
      /*tolerance=*/0.0157651,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteUInt8);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridUInt8NoCifgNoPphlNoPrjAsymTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_config,
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_buffers,
      /*tolerance=*/0.0157651,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteUInt8,
      /*asymmetric_quantize_inputs=*/true);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmFloatNoCifgNoPphlNoPrjBatchMajorTest) {
  tflite::testing::TestUnidirectionalSequenceLstmFloat(
      &tflite::testing::lstm_no_cifg_no_peephole_no_proj_config,
      /*tolerance=*/1e-5,
      /*input_output_batch_major=*/true);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmFloatCifgPphlNoPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmFloat(
      &tflite::testing::lstm_cifg_peephole_no_proj_config,
      /*tolerance=*/1e-5,
      /*input_output_batch_major=*/false);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridInt8CifgPphlNoPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_cifg_peephole_no_proj_config,
      &tflite::testing::lstm_cifg_peephole_no_proj_buffers,
      /*tolerance=*/0.03573,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteInt8);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridInt8CifgPphlNoPrjAsymTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_cifg_peephole_no_proj_config,
      &tflite::testing::lstm_cifg_peephole_no_proj_buffers,
      /*tolerance=*/0.03573,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteInt8,
      /*asymmetric_quantize_inputs=*/true);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridUInt8CifgPphlNoPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_cifg_peephole_no_proj_config,
      &tflite::testing::lstm_cifg_peephole_no_proj_buffers,
      /*tolerance=*/0.03573,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteUInt8);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridUInt8CifgPphlNoPrjAsymTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_cifg_peephole_no_proj_config,
      &tflite::testing::lstm_cifg_peephole_no_proj_buffers,
      /*tolerance=*/0.03573,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteUInt8,
      /*asymmetric_quantize_inputs=*/true);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmFloatNoCifgPphlPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmFloat(
      &tflite::testing::lstm_no_cifg_peephole_proj_config,
      /*tolerance=*/1e-5,
      /*input_output_batch_major=*/false);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridInt8NoCifgPphlPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_no_cifg_peephole_proj_config,
      &tflite::testing::lstm_no_cifg_peephole_proj_buffers,
      /*tolerance=*/0.00467,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteInt8);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmHybridUInt8NoCifgPphlPrjTest) {
  tflite::testing::TestUnidirectionalSequenceLstmHybrid(
      &tflite::testing::lstm_no_cifg_peephole_proj_config,
      &tflite::testing::lstm_no_cifg_peephole_proj_buffers,
      /*tolerance=*/0.00467,
      /*input_output_batch_major=*/false,
      /*weight_type=*/kTfLiteUInt8);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmFloatNoCifgPphlPrjBiasTest) {
  tflite::testing::TestUnidirectionalSequenceLstmFloat(
      &tflite::testing::lstm_no_cifg_peephole_proj_bias_config,

      /*tolerance=*/1e-5,
      /*input_output_batch_major=*/false);
}

TF_LITE_MICRO_TEST(UndrctnlSqncLstmFloatCifgPphlNoPrjLayerNormTest) {
  tflite::testing::TestUnidirectionalSequenceLstmFloat(
      &tflite::testing::cifg_peephole_no_proj_config_layer_norm,

      /*tolerance=*/1e-5,
      /*input_output_batch_major=*/false);
}

#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_END
