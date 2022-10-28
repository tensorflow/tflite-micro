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
#include "tensorflow/lite/micro/kernels/lstm_eval.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {
// Model Constants
constexpr int kInputDimension = 2;
constexpr int kStateDimension = 2;
constexpr int kBatchSize = 2;
constexpr int kTimeSteps = 3;
constexpr int kInputSize = kBatchSize * kTimeSteps * kInputDimension;
constexpr int kOutputSize = kBatchSize * kTimeSteps * kStateDimension;
constexpr int kGateOutputSize = kBatchSize * kStateDimension;

// Number of tensors for the LSTM kernel. 0 input; 1-12 gate parameters; 13-14
// states; 15 output
constexpr int kTensorsNum = 16;

constexpr TfLiteLSTMParams kModelSettings = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/10, /*.proj_clip=*/3,
    /*.kernel_type=*/kTfLiteLSTMFullKernel,
    /*.asymmetric_quantize_inputs=*/true};

// batch one using repeated data; batch two mimics
// random input
constexpr float kInputData[kInputSize] = {
    0.2,   0.3,  0.2,  0.3,  0.2,  0.3,   // batch one
    -0.98, 0.62, 0.01, 0.99, 0.49, -0.32  // batch two
};

// The expected model output after kTimeSteps using the fixed input and
// parameters
constexpr float kExpectedOutput[kOutputSize] = {
    0.26455893,      0.26870455,      0.47935803,
    0.47937014,      0.58013272,      0.58013278,  // batch1
    -1.41184672e-3f, -1.43329117e-5f, 0.46887168,
    0.46891281,      0.50054074,      0.50054148  // batch2
};

// Test Settings
constexpr float kTestFloatTolerance = 1e-6f;

// Struct that holds the weight/bias information for a standard gate (i.e. no
// modification such as layer normalization, peephole, etc.)
template <typename WeightType, typename BiasType>
struct GateParameters {
  WeightType activation_weight[kStateDimension * kInputDimension];
  WeightType recurrent_weight[kStateDimension * kStateDimension];
  BiasType fused_bias[kStateDimension];
  // Quantized model folded the zero point of activations into biases:
  // bias + zero_point * weight.
  BiasType activation_zp_folded_bias[kStateDimension];
  BiasType recurrent_zp_folded_bias[kStateDimension];
};

// Parameters for different gates
// negative large weights for forget gate to make it really forget
constexpr GateParameters<float, float> kForgetGateParameters = {
    /*.activation_weight=*/{-10, -10, -20, -20},
    /*.recurrent_weight=*/{-10, -10, -20, -20},
    /*.fused_bias=*/{1, 2}};
// positive large weights for input gate to make it really remember
constexpr GateParameters<float, float> kInputGateParameters = {
    /*.activation_weight=*/{10, 10, 20, 20},
    /*.recurrent_weight=*/{10, 10, 20, 20},
    /*.fused_bias=*/{-1, -2},
    /*activation_zp_folded_bias=*/{0, 0},
    /*recurrent_zp_folded_bias=*/{0, 0}};
// all ones to test the behavior of tanh at normal range (-1,1)
constexpr GateParameters<float, float> kCellGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0},
    /*activation_zp_folded_bias=*/{0, 0},
    /*recurrent_zp_folded_bias=*/{0, 0}};
// all ones to test the behavior of sigmoid at normal range (-1. 1)
constexpr GateParameters<float, float> kOutputGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0},
    /*activation_zp_folded_bias=*/{0, 0},
    /*recurrent_zp_folded_bias=*/{0, 0}};

// Class that holds all the tensors for evaluation
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType>
class ModelContents {
 public:
  ModelContents(const GateParameters<WeightType, BiasType> forget_gate_params,
                const GateParameters<WeightType, BiasType> input_gate_params,
                const GateParameters<WeightType, BiasType> cell_gate_params,
                const GateParameters<WeightType, BiasType> output_gate_params)
      : forget_gate_params_(forget_gate_params),
        input_gate_params_(input_gate_params),
        cell_gate_params_(cell_gate_params),
        output_gate_params_(output_gate_params) {
    InitializeTensors();
  }

  // Provide interface to set the input tensor values for flexible testing
  void SetInputTensorData(const ActivationType* data) {
    std::memcpy(input_, data, kInputSize * sizeof(ActivationType));
    SetTensor(0, input_, input_size_);
  }

  TfLiteEvalTensor* GetTensor(int tensor_index) {
    return tensors_ + tensor_index;
  }
  const ActivationType* GetHiddenState() const { return hidden_state_; }
  const CellType* GetCellState() const { return cell_state_; }
  const ActivationType* GetOutput() const { return output_; }

  CellType* ScratchBuffers() { return scratch_buffers_; }

 protected:
  GateParameters<WeightType, BiasType> forget_gate_params_;
  GateParameters<WeightType, BiasType> input_gate_params_;
  GateParameters<WeightType, BiasType> cell_gate_params_;
  GateParameters<WeightType, BiasType> output_gate_params_;

  // Not const since IntArrayFromInts takes int *; the first element of the
  // array must be the size of the array
  int input_size_[4] = {3, kBatchSize, kTimeSteps, kInputDimension};
  int output_size_[4] = {3, kBatchSize, kTimeSteps, kStateDimension};
  int activation_weight_size_[3] = {2, kStateDimension, kInputDimension};
  int recurrent_weight_size_[3] = {2, kStateDimension, kStateDimension};
  int bias_size_[3] = {2, kBatchSize, kStateDimension};

  // 0 input; 1-12 gate parameters; 13-14 states; 15 output
  TfLiteEvalTensor tensors_[kTensorsNum];

  // states are initialized to zero
  ActivationType hidden_state_[kGateOutputSize] = {0};
  CellType cell_state_[kGateOutputSize] = {0};
  // input is defined in the ModelContent (const across all derived models)
  ActivationType input_[kOutputSize] = {0};
  ActivationType output_[kOutputSize] = {0};
  // scratch buffers (4)
  CellType scratch_buffers_[4 * kGateOutputSize] = {0};

  void InitializeTensors() {
    // Input Tensor
    SetTensor(0, input_, input_size_);
    // Forget Gate Tensors
    SetTensor(1, forget_gate_params_.activation_weight,
              activation_weight_size_);
    SetTensor(2, forget_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(3, forget_gate_params_.fused_bias, bias_size_);
    // Input Gate Tensors
    SetTensor(4, input_gate_params_.activation_weight, activation_weight_size_);
    SetTensor(5, input_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(6, input_gate_params_.fused_bias, bias_size_);
    // Cell Gate Tensors
    SetTensor(7, cell_gate_params_.activation_weight, activation_weight_size_);
    SetTensor(8, cell_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(9, cell_gate_params_.fused_bias, bias_size_);
    // Output Gate Tensors
    SetTensor(10, output_gate_params_.activation_weight,
              activation_weight_size_);
    SetTensor(11, output_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(12, output_gate_params_.fused_bias, bias_size_);
    // State Tensors
    SetTensor(13, hidden_state_, output_size_);
    SetTensor(14, cell_state_, output_size_);
    // Output Tensor
    SetTensor(15, output_, output_size_);
  }

  template <typename T>
  void SetTensor(const int index, const T* data, int* dims) {
    tensors_[index].data.data = const_cast<T*>(data);
    tensors_[index].dims = IntArrayFromInts(dims);
    tensors_[index].type = typeToTfLiteType<T>();
  }
};

// A struct that holds quantization parameters for a LSTM Tensor
struct TensorQuantizationParameters {
  const float scale;
  const float zero_point;
  const bool symmetry;
};

struct GateQuantizationParameters {
  TensorQuantizationParameters activation_weight;
  TensorQuantizationParameters recurrent_weight;
  TensorQuantizationParameters bias;
};

// A struct that holds the quantization settings for the model
struct ModelQuantizationParameters {
  TfLiteType activation_type;
  TfLiteType cell_type;
  TfLiteType bias_type;
  float nonlinear_activation_input_scale;
  float nonlinear_activation_output_scale;
  // Quantization parameters for input/output
  TensorQuantizationParameters input_quantization_parameters;
  TensorQuantizationParameters output_quantization_parameters;
  // Quantization parameters for internal states
  TensorQuantizationParameters hidden_quantization_parameters;
  TensorQuantizationParameters cell_quantization_parameters;
  // Quantization parameters for gates
  GateQuantizationParameters forget_gate_quantization_parameters;
  GateQuantizationParameters input_gate_quantization_parameters;
  GateQuantizationParameters cell_gate_quantization_parameters;
  GateQuantizationParameters output_gate_quantization_parameters;
};

constexpr ModelQuantizationParameters kInt8QuantizationSettings = {
    /*activation_type=*/kTfLiteInt8,
    /*cell_type=*/kTfLiteInt16,
    /*bias_type=*/kTfLiteInt32,
    /*nonlinear_input_scale=*/0.00024414062,   // std::pow(2.0f, -12.0f)
    /*nonlinear_output_scale=*/0.00003051757,  // std::pow(2.0f, -15.0f)
    /*Input=*/{/*scale=*/0.00784313725490196, /*zp=*/0, /*symmetry=*/false},
    /*output=*/{/*scale=*/0.00392156862745098, /*zp=*/-26, /*symmetry=*/false},
    /*hidden=*/{/*scale=*/0.00392156862745098, /*zp=*/-26, /*symmetry=*/false},
    /*cell=*/{/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true},

    /*forget_gate=*/
    {/*activation_weight=*/{/*scale=*/0.15748031496062992, /*zp=*/0,
                            /*symmetry=*/true},
     /*recurrent_weight*/
     {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
     /*bias=*/{/*scale=*/0.0012351397251814111, /*zp=*/0, /*symmetry=*/true}},

    /*input_gate=*/
    {/*activation_weight=*/{/*scale=*/0.15748031496062992, /*zp=*/0,
                            /*symmetry=*/true},
     /*recurrent_weight*/
     {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
     /*bias=*/{/*scale=*/0.0012351397251814111, /*zp=*/0, /*symmetry=*/true}},

    /*cell_gate=*/
    {/*activation_weight=*/{/*scale=*/0.007874015748031496, /*zp=*/0,
                            /*symmetry=*/true},
     /*recurrent_weight*/
     {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
     /*bias=*/{/*scale=*/6.175698625907056e-5f, /*zp=*/0, /*symmetry=*/true}},

    /*output_gate=*/
    {/*activation_weight=*/{/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true},
     /*recurrent_weight*/ {/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true},
     /*bias=*/{/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true}}};

// A function that converts floating point gate parameters to the
// corresponding quantized version
template <typename WeightType, typename BiasType>
GateParameters<WeightType, BiasType> QuantizeGateParameters(
    const GateParameters<float, float>& gate_parameters,
    const TensorQuantizationParameters& input_quantization_params,
    const TensorQuantizationParameters& output_quantization_params,
    const GateQuantizationParameters& gate_quantization_params) {
  GateParameters<WeightType, BiasType> quantized_gate_params;
  // Quantize the activation weight
  tflite::SymmetricQuantize(gate_parameters.activation_weight,
                            quantized_gate_params.activation_weight,
                            kStateDimension * kInputDimension,
                            gate_quantization_params.activation_weight.scale);
  // Quantize the recurrent weight
  tflite::SymmetricQuantize(gate_parameters.recurrent_weight,
                            quantized_gate_params.recurrent_weight,
                            kStateDimension * kStateDimension,
                            gate_quantization_params.recurrent_weight.scale);
  // Quantize the bias
  tflite::SymmetricQuantize(gate_parameters.fused_bias,
                            quantized_gate_params.fused_bias, kStateDimension,
                            gate_quantization_params.bias.scale);

  // Copy the bias values to prepare zero_point folded bias precomputation (bias
  // has same scale as input_scale*input_weight_scale)
  std::memcpy(quantized_gate_params.activation_zp_folded_bias,
              quantized_gate_params.fused_bias,
              kStateDimension * sizeof(BiasType));
  // Pre-calculate bias - zero_point * weight.
  tflite::tensor_utils::MatrixScalarMultiplyAccumulate(
      quantized_gate_params.activation_weight,
      -input_quantization_params.zero_point, kStateDimension, kInputDimension,
      quantized_gate_params.activation_zp_folded_bias);

  tflite::tensor_utils::MatrixScalarMultiplyAccumulate(
      quantized_gate_params.recurrent_weight,
      -output_quantization_params.zero_point, kStateDimension, kStateDimension,
      quantized_gate_params.recurrent_zp_folded_bias);

  return quantized_gate_params;
}

const GateParameters<int8_t, int32_t> kIint8ForgetGateParams =
    QuantizeGateParameters<int8_t, int32_t>(
        kForgetGateParameters,
        kInt8QuantizationSettings.input_quantization_parameters,
        kInt8QuantizationSettings.output_quantization_parameters,
        kInt8QuantizationSettings.forget_gate_quantization_parameters);

const GateParameters<int8_t, int32_t> kIint8InputGateParams =
    QuantizeGateParameters<int8_t, int32_t>(
        kInputGateParameters,
        kInt8QuantizationSettings.input_quantization_parameters,
        kInt8QuantizationSettings.output_quantization_parameters,
        kInt8QuantizationSettings.input_gate_quantization_parameters);

const GateParameters<int8_t, int32_t> kIint8CellGateParams =
    QuantizeGateParameters<int8_t, int32_t>(
        kCellGateParameters,
        kInt8QuantizationSettings.input_quantization_parameters,
        kInt8QuantizationSettings.output_quantization_parameters,
        kInt8QuantizationSettings.cell_gate_quantization_parameters);

const auto kInt8OutputGateParams = QuantizeGateParameters<int8_t, int32_t>(
    kOutputGateParameters,
    kInt8QuantizationSettings.input_quantization_parameters,
    kInt8QuantizationSettings.output_quantization_parameters,
    kInt8QuantizationSettings.output_gate_quantization_parameters);

// Class that contains all the information to run quantized LSTM inference
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType>
class QuantizedModelContents
    : public ModelContents<ActivationType, WeightType, BiasType, CellType> {
 public:
  QuantizedModelContents(
      const ModelQuantizationParameters quantization_settings,
      const GateParameters<WeightType, BiasType> quantized_forget_gate_params,
      const GateParameters<WeightType, BiasType> quantized_input_gate_params,
      const GateParameters<WeightType, BiasType> quantized_cell_gate_params,
      const GateParameters<WeightType, BiasType> quantized_output_gate_params)
      : ModelContents<ActivationType, WeightType, BiasType, CellType>(
            quantized_forget_gate_params, quantized_input_gate_params,
            quantized_cell_gate_params, quantized_output_gate_params),
        quantization_settings_(quantization_settings) {
    // Setup the IntegerLstmParameter
    AssembleEvalualtionParams();
  }

  const IntegerLstmParameter& GetEvaluationParameters() const {
    return evaluation_params_;
  }

 private:
  // Quantization settings for every tensor inside the model
  const ModelQuantizationParameters quantization_settings_;
  // All the information that required to invoke the quantized kernel
  IntegerLstmParameter evaluation_params_;

  // Set the input tensor, quantized version
  void QuantizeInputTensorData(const float* data) {
    ActivationType quantized_input_data[kInputSize];
    Quantize(data, quantized_input_data, kInputSize,
             quantization_settings_.input_quantization_parameters.scale,
             quantization_settings_.input_quantization_parameters.zero_point);
    SetInputTensorData(quantized_input_data);
  }

  void AssembleEvalualtionParams() {
    double effective_scale;
    // Forget Gate
    effective_scale =
        quantization_settings_.input_quantization_parameters.scale *
        quantization_settings_.forget_gate_quantization_parameters
            .activation_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(effective_scale,
                       &evaluation_params_.effective_input_to_forget_scale_a,
                       &evaluation_params_.effective_input_to_forget_scale_b);
    effective_scale =
        quantization_settings_.output_quantization_parameters.scale *
        quantization_settings_.forget_gate_quantization_parameters
            .recurrent_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(
        effective_scale,
        &evaluation_params_.effective_recurrent_to_forget_scale_a,
        &evaluation_params_.effective_recurrent_to_forget_scale_b);
    // Set effective bias
    evaluation_params_.input_to_forget_effective_bias =
        this->forget_gate_params_.activation_zp_folded_bias;
    evaluation_params_.recurrent_to_forget_effective_bias =
        this->forget_gate_params_.recurrent_zp_folded_bias;

    // input gate
    effective_scale =
        quantization_settings_.input_quantization_parameters.scale *
        quantization_settings_.input_gate_quantization_parameters
            .activation_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(effective_scale,
                       &evaluation_params_.effective_input_to_input_scale_a,
                       &evaluation_params_.effective_input_to_input_scale_b);
    effective_scale =
        quantization_settings_.output_quantization_parameters.scale *
        quantization_settings_.input_gate_quantization_parameters
            .recurrent_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(
        effective_scale,
        &evaluation_params_.effective_recurrent_to_input_scale_a,
        &evaluation_params_.effective_recurrent_to_input_scale_b);
    // Set effective bias
    evaluation_params_.input_to_input_effective_bias =
        this->input_gate_params_.activation_zp_folded_bias;
    evaluation_params_.recurrent_to_input_effective_bias =
        this->input_gate_params_.recurrent_zp_folded_bias;

    // cell gate
    effective_scale =
        quantization_settings_.input_quantization_parameters.scale *
        quantization_settings_.cell_gate_quantization_parameters
            .activation_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(effective_scale,
                       &evaluation_params_.effective_input_to_cell_scale_a,
                       &evaluation_params_.effective_input_to_cell_scale_b);
    effective_scale =
        quantization_settings_.output_quantization_parameters.scale *
        quantization_settings_.cell_gate_quantization_parameters
            .recurrent_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(effective_scale,
                       &evaluation_params_.effective_recurrent_to_cell_scale_a,
                       &evaluation_params_.effective_recurrent_to_cell_scale_b);
    // Set effective bias
    evaluation_params_.input_to_cell_effective_bias =
        this->cell_gate_params_.activation_zp_folded_bias;
    evaluation_params_.recurrent_to_cell_effective_bias =
        this->cell_gate_params_.recurrent_zp_folded_bias;

    // output gate
    effective_scale =
        quantization_settings_.input_quantization_parameters.scale *
        quantization_settings_.output_gate_quantization_parameters
            .activation_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(effective_scale,
                       &evaluation_params_.effective_input_to_output_scale_a,
                       &evaluation_params_.effective_input_to_output_scale_b);
    effective_scale =
        quantization_settings_.output_quantization_parameters.scale *
        quantization_settings_.output_gate_quantization_parameters
            .recurrent_weight.scale /
        quantization_settings_.nonlinear_activation_input_scale;
    QuantizeMultiplier(
        effective_scale,
        &evaluation_params_.effective_recurrent_to_output_scale_a,
        &evaluation_params_.effective_recurrent_to_output_scale_b);
    // Set effective bias
    evaluation_params_.input_to_output_effective_bias =
        this->output_gate_params_.activation_zp_folded_bias;
    evaluation_params_.recurrent_to_output_effective_bias =
        this->output_gate_params_.recurrent_zp_folded_bias;

    // hidden state (no projection, output is the hidden state)
    effective_scale =
        quantization_settings_.nonlinear_activation_input_scale *
        quantization_settings_.nonlinear_activation_input_scale /
        quantization_settings_.output_quantization_parameters.scale;
    QuantizeMultiplier(effective_scale,
                       &evaluation_params_.effective_hidden_scale_a,
                       &evaluation_params_.effective_hidden_scale_b);
    evaluation_params_.hidden_zp =
        quantization_settings_.output_quantization_parameters.zero_point;

    // cell state. Note, cell_scale is actually not a scale. 2^-cell_scale is
    // the true scale for cell
    tflite::CheckedLog2(
        quantization_settings_.cell_quantization_parameters.scale,
        &evaluation_params_.cell_scale);
  }
};

template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

void TestGateOutputFloat(const GateParameters<float, float>& gate_params,
                         TfLiteFusedActivation activation_type,
                         const float* input_data, const float* hidden_state,
                         const float* expected_vals) {
  float gate_output[kGateOutputSize] = {0};
  tflite::lstm_internal::CalculateLstmGateFloat(
      input_data, gate_params.activation_weight,
      /*aux_input=*/nullptr, /*aux_input_to_gate_weights*/ nullptr,
      hidden_state, gate_params.recurrent_weight,
      /*cell_state=*/nullptr, /*cell_to_gate_weights=*/nullptr,
      /*layer_norm_coefficients=*/nullptr, gate_params.fused_bias, kBatchSize,
      kInputDimension, kInputDimension, kStateDimension, kStateDimension,
      /*activation=*/activation_type, gate_output,
      /*is_input_all_zeros=*/false,
      /*is_aux_input_all_zeros=*/true);
  ValidateResultGoldens(expected_vals, gate_output, kGateOutputSize,
                        kTestFloatTolerance);
}

// TODO(rewu): Clean up the input parameters, which requires refactor
// IntegerLstmParameter
template <typename ActivationType, typename BiasType, typename CellType>
void TestGateOutputQuantized(
    const GateParameters<int8_t, BiasType>& gate_params,
    const ModelQuantizationParameters& quantization_settings,
    int32_t effective_input_to_gate_scale_a,
    int32_t effective_input_to_gate_scale_b,
    int32_t effective_recurrent_to_gate_scale_a,
    int32_t effective_recurrent_to_gate_scale_b,
    TfLiteFusedActivation nonlinear_type, const float* input_data,
    const float* hidden_state, const float* expected_vals, float tolerance) {
  // Quantize the  floating point input
  ActivationType quantized_input[kInputSize];
  Quantize(input_data, quantized_input, kInputSize,
           quantization_settings.input_quantization_parameters.scale,
           quantization_settings.input_quantization_parameters.zero_point);
  // Quantize the  floating point hidden state
  ActivationType quantized_hidden_state[kGateOutputSize];
  Quantize(hidden_state, quantized_hidden_state, kGateOutputSize,
           quantization_settings.output_quantization_parameters.scale,
           quantization_settings.output_quantization_parameters.zero_point);

  CellType gate_output[kGateOutputSize] = {0};
  BiasType scratch_buffer[kGateOutputSize];

  tflite::lstm_internal::CalculateLstmGateInteger8x8_16(
      // Input and weights
      quantized_input, gate_params.activation_weight,
      gate_params.activation_zp_folded_bias, effective_input_to_gate_scale_a,
      effective_input_to_gate_scale_b,
      // Output state and weights
      quantized_hidden_state, gate_params.activation_weight,
      gate_params.recurrent_zp_folded_bias, effective_recurrent_to_gate_scale_a,
      effective_recurrent_to_gate_scale_b,
      // Cell state and weights
      nullptr, nullptr, 0, 0,
      // Layer normalization parameters (layer norm LSTM)
      nullptr, nullptr, 0, 0, 0,
      // Array sizes
      kBatchSize, kInputDimension, kStateDimension, kStateDimension,
      nonlinear_type,
      // Output
      gate_output,
      // Parameters for performance optimizations
      // Scratch arrays
      scratch_buffer);

  float gate_output_float[kGateOutputSize];
  Dequantize(gate_output, kGateOutputSize,
             quantization_settings.nonlinear_activation_output_scale, 0,
             gate_output_float);

  ValidateResultGoldens(expected_vals, gate_output_float, kGateOutputSize,
                        tolerance);
}

void TestOneStepLSTMFloat(const float* input_data,
                          const float* expected_hidden_state,
                          const float* expected_cell_state, float* hidden_state,
                          float* cell_state, float* output) {
  // scratch buffers
  float forget_gate_scratch[kGateOutputSize];
  float input_gate_scratch[kGateOutputSize];
  float cell_gate_scratch[kGateOutputSize];
  float output_gate_scratch[kGateOutputSize];

  tflite::lstm_internal::LstmStepFloat(
      input_data, kInputGateParameters.activation_weight,
      kForgetGateParameters.activation_weight,
      kCellGateParameters.activation_weight,
      kOutputGateParameters.activation_weight,
      /*aux_input_ptr=*/nullptr, /*aux_input_to_input_weights_ptr=*/nullptr,
      /*aux_input_to_forget_weights_ptr=*/nullptr,
      /*aux_input_to_cell_weights_ptr=*/nullptr,
      /*aux_input_to_output_weights_ptr=*/nullptr,
      kInputGateParameters.recurrent_weight,
      kForgetGateParameters.recurrent_weight,
      kCellGateParameters.recurrent_weight,
      kOutputGateParameters.recurrent_weight,
      /*cell_to_input_weights_ptr=*/nullptr,
      /*cell_to_forget_weights_ptr=*/nullptr,
      /*cell_to_output_weights_ptr=*/nullptr,
      /*input_layer_norm_coefficients_ptr=*/nullptr,
      /*forget_layer_norm_coefficients_ptr=*/nullptr,
      /*cell_layer_norm_coefficients_ptr=*/nullptr,
      /*output_layer_norm_coefficients_ptr=*/nullptr,
      kInputGateParameters.fused_bias, kForgetGateParameters.fused_bias,
      kCellGateParameters.fused_bias, kOutputGateParameters.fused_bias,
      /*projection_weights_ptr=*/nullptr, /*projection_bias_ptr=*/nullptr,
      &kModelSettings, kBatchSize, kStateDimension, kInputDimension,
      kInputDimension, kStateDimension,
      /*output_batch_leading_dim=*/0, hidden_state, cell_state,
      input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
      output_gate_scratch, output);

  ValidateResultGoldens(expected_hidden_state, hidden_state, kGateOutputSize,
                        kTestFloatTolerance);
  ValidateResultGoldens(expected_cell_state, cell_state, kGateOutputSize,
                        kTestFloatTolerance);
}

struct GateOutputCheckData {
  const float input_data[kInputSize] = {
      0.2, 0.3,    // batch1
      -0.98, 0.62  // batch2
  };
  const float hidden_state[kGateOutputSize] = {
      -0.1, 0.2,  // batch1
      -0.3, 0.5   // batch2
  };
  // Use the forget gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[-10,-10],[-20,-20]][0.2,
  // +[[-10,-10],[-20,-20]][-0.1, 0.2]+[1,2]) = sigmoid([-5,-10]) =
  // [6.69285092e-03, 4.53978687e-05] (Batch1)
  // Similarly, we have [0.93086158 0.9945137 ] for batch 2
  const float expected_forget_gate_output[kGateOutputSize] = {
      6.69285092e-3f, 4.53978687e-5f, 0.93086158, 0.9945137};

  // Use the input gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[10,10],[20,20]][0.2, 0.3]
  // +[[10,10],[20,20]][-0.1, 0.2]+[-1,-2]) = sigmoid([5,10]) =
  // [0.99330715, 0.9999546]
  // Similarly, we have [0.06913842 0.0054863 ] for batch 2
  const float expected_input_gate_output[kGateOutputSize] = {
      0.99330715, 0.9999546, 0.06913842, 0.0054863};

  // Use the output gate parameters to test normnal gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = sigmoid([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  // Similarly, we have [[0.46008512 0.46008512]] for batch 2
  const float expected_output_gate_output[kGateOutputSize] = {
      0.6456563062257954, 0.6456563062257954, 0.46008512, 0.46008512};

  // Use the cell(modulation) gate parameters to tanh output
  // output = tanh(W_i*i+W_h*h+b) = tanh([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = tanh([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  // Similarly, we have [-0.1586485 -0.1586485] for batch 2
  const float expected_cell_gate_output[kGateOutputSize] = {
      0.5370495669980353, 0.5370495669980353, -0.1586485, -0.1586485};
};

const std::unique_ptr<tflite::testing::GateOutputCheckData> kGateOutputData(
    new GateOutputCheckData);

const std::unique_ptr<
    tflite::testing::ModelContents<float, float, float, float>>
    kFloatModelContent(
        new tflite::testing::ModelContents<float, float, float, float>(
            tflite::testing::kForgetGateParameters,
            tflite::testing::kInputGateParameters,
            tflite::testing::kCellGateParameters,
            tflite::testing::kOutputGateParameters));

const std::unique_ptr<QuantizedModelContents<int8_t, int8_t, int32_t, int16_t>>
    kInt8ModelContent(
        new QuantizedModelContents<int8_t, int8_t, int32_t, int16_t>(
            kInt8QuantizationSettings, kIint8ForgetGateParams,
            kIint8InputGateParams, kIint8CellGateParams,
            kInt8OutputGateParams));

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
TF_LITE_MICRO_TEST(CheckGateOutputFloat) {
  // Forget gate
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kForgetGateParameters, kTfLiteActSigmoid,
      tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_forget_gate_output);
  // Input gate
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kInputGateParameters, kTfLiteActSigmoid,
      tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_input_gate_output);
  // output gate
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kOutputGateParameters, kTfLiteActSigmoid,
      tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_output_gate_output);
  // cell (modulation) gate
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kCellGateParameters,
      tflite::testing::kModelSettings.activation,
      tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_cell_gate_output);
}

TF_LITE_MICRO_TEST(CheckGateOutputInt8) {
  auto& evaluation_params =
      tflite::testing::kInt8ModelContent->GetEvaluationParameters();
  float tolerance;

  // Forget Gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t>(
      tflite::testing::kIint8ForgetGateParams,
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_forget_scale_a,
      evaluation_params.effective_input_to_forget_scale_b,
      evaluation_params.effective_recurrent_to_forget_scale_a,
      evaluation_params.effective_recurrent_to_forget_scale_b,
      kTfLiteActSigmoid, tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_forget_gate_output, tolerance);

  // Input Gate
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t>(
      tflite::testing::kIint8InputGateParams,
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_input_scale_a,
      evaluation_params.effective_input_to_input_scale_b,
      evaluation_params.effective_recurrent_to_input_scale_a,
      evaluation_params.effective_recurrent_to_input_scale_b, kTfLiteActSigmoid,
      tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_input_gate_output, tolerance);

  // Output Gate
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t>(
      tflite::testing::kInt8OutputGateParams,
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_output_scale_a,
      evaluation_params.effective_input_to_output_scale_b,
      evaluation_params.effective_recurrent_to_output_scale_a,
      evaluation_params.effective_recurrent_to_output_scale_b,
      kTfLiteActSigmoid, tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_output_gate_output, tolerance);

  // Cell Gate (tanh activation)
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t>(
      tflite::testing::kIint8CellGateParams,
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_cell_scale_a,
      evaluation_params.effective_input_to_cell_scale_b,
      evaluation_params.effective_recurrent_to_cell_scale_a,
      evaluation_params.effective_recurrent_to_cell_scale_b,
      tflite::testing::kModelSettings.activation,
      tflite::testing::kGateOutputData->input_data,
      tflite::testing::kGateOutputData->hidden_state,
      tflite::testing::kGateOutputData->expected_cell_gate_output, tolerance);
}

TF_LITE_MICRO_TEST(CheckCellUpdate) {
  float cell_state[] = {0.1, 0.2, 0.3, 0.4};
  float forget_gate[] = {0.2, 0.5, 0.3, 0.6};
  const float input_gate[] = {0.8, 0.9, 0.6, 0.7};
  const float cell_gate[] = {-0.3, 0.8, 0.1, 0.2};

  // Cell = forget_gate*cell + input_gate*cell_gate
  // = [0.02, 0.1, 0.09, 0.24] + [-0.24, 0.72 , 0.06, 0.14 ] = [-0.22, 0.82]
  const float expected_cell_vals[] = {-0.22, 0.82, 0.15, 0.38};

  tflite::lstm_internal::UpdateLstmCellFloat(
      tflite::testing::kBatchSize, tflite::testing::kStateDimension, cell_state,
      input_gate, forget_gate, cell_gate, /*use_cifg=*/false,
      /*clip=*/tflite::testing::kModelSettings.cell_clip);

  tflite::testing::ValidateResultGoldens(expected_cell_vals, cell_state,
                                         tflite::testing::kGateOutputSize,
                                         tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckOutputCalculation) {
  // -1 and 5 for different tanh behavior (normal, saturated)
  const float cell_state[] = {-1, 5, 0.1, -3};
  const float output_gate[] = {0.2, 0.5, -0.8, 0.9};
  // If no projection layer, hidden state dimension == output dimension ==
  // cell state dimension
  float output[tflite::testing::kGateOutputSize];
  float scratch[tflite::testing::kGateOutputSize];

  tflite::lstm_internal::CalculateLstmOutputFloat(
      tflite::testing::kBatchSize, tflite::testing::kStateDimension,
      tflite::testing::kStateDimension, cell_state, output_gate, kTfLiteActTanh,
      nullptr, nullptr, 0, output, scratch);

  // Output state generate the output and copy it to the hidden state
  // tanh(cell_state) * output_gate =
  // [-0.15231883,  0.4999546 , -0.0797344 , -0.89554928]
  float expected_output_vals[] = {-0.15231883, 0.4999546, -0.0797344,
                                  -0.89554928};
  tflite::testing::ValidateResultGoldens(expected_output_vals, output,
                                         tflite::testing::kGateOutputSize,
                                         tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTM) {
  // initialize states as zero arrays (as in real cases)
  float hidden_state[] = {
      0, 0,  // batch1
      0, 0   // batch2
  };
  float cell_state[] = {
      0, 0,  // batch1
      0, 0   // batch2
  };
  // initialize State
  float output[tflite::testing::kGateOutputSize];

  // Previous hidden_state (h): [0. 0.]
  // Previous cell_state (cell): [0. 0.]
  // Forget gate output: [0.01798621 0.00033535]
  // Long term state after forgetting: [0. 0.]
  // Input gate output: [0.98201379 0.99966465]
  // Modulation gate output: [0.46211716 0.46211716]
  // Gated input: [0.45380542 0.46196219]
  // Long term state after adding input: [0.45380542 0.46196219]
  // Output_gate_output: [0.62245933 0.62245933]
  // Gated output: [0.26455893 0.26870455]
  // Current hidden_state (h): [0.26455893 0.26870455]
  // Current cell_state (cell): [0.45380542 0.46196219] (batch1)
  const float step1_input_data[] = {
      0.2, 0.3,    // batch1
      -0.98, 0.62  // batch2
  };
  const float step1_expected_hidden_state[] = {
      0.26455893, 0.26870455,           // batch1
      -1.41184672e-3f, -1.43329117e-5f  // batch2
  };
  const float step1_expected_cell_state[] = {
      0.45380542, 0.46196219,           // batch1
      -3.43550167e-3f, -3.48766956e-5f  // batch2
  };
  tflite::testing::TestOneStepLSTMFloat(
      step1_input_data, step1_expected_hidden_state, step1_expected_cell_state,
      hidden_state, cell_state, output);

  // Previous hidden_state (h): [0.26455893 0.26870455]
  // Previous cell_state (cell): [0.45380542 0.46196219]
  // Forget gate output: [8.8480948e-05 7.8302637e-09]
  // Long term state after forgetting: [4.01531339e-05 3.61728574e-09]
  // Input gate output: [0.99991152 0.99999999]
  // Modulation gate output: [0.77521391 0.77521391]
  // Gated input: [0.77514532 0.7752139 ]
  // Long term state after adding input: [0.77518547 0.77521391]
  // Output_gate_output: [0.7375481 0.7375481]
  // Gated output: [0.47935803 0.47937014]
  // Current hidden_state (h): [0.47935803 0.47937014]
  // Current cell_state (cell): [0.77518547 0.77521391] (batch1)
  const float step2_input_data[] = {
      0.2, 0.3,   // batch1
      0.01, 0.99  // batch2
  };
  const float step2_expected_hidden_state[] = {
      0.47935803, 0.47937014,  // batch1
      0.46887168, 0.46891281   // batch2
  };
  const float step2_expected_cell_state[] = {
      0.77518547, 0.77521391,  // batch1
      0.76089886, 0.76099453   // batch2
  };
  tflite::testing::TestOneStepLSTMFloat(
      step2_input_data, step2_expected_hidden_state, step2_expected_cell_state,
      hidden_state, cell_state, output);

  // Previous hidden_state (h): [0.47935803 0.47937014]
  // Previous cell_state (cell): [0.77518547 0.77521391]
  // Forget gate output: [1.25637118e-06 1.57847251e-12]
  // Long term state after forgetting: [9.73920683e-07 1.22365384e-12]
  // Input gate output: [0.99999874 1.        ]
  // Modulation gate output: [0.8974053 0.8974053]
  // Gated input: [0.89740417 0.8974053 ]
  // Long term state after adding input: [0.89740515 0.8974053 ]
  // Output_gate_output: [0.81133808 0.81133808]
  // Gated output: [0.58013272 0.58013278]
  // Current hidden_state (h): [0.58013272 0.58013278]
  // Current cell_state (cell): [0.89740515 0.8974053 ] (batch1)
  const float step3_input_data[] = {
      0.2, 0.3,    // batch1
      0.49, -0.32  // batch2
  };
  const float step3_expected_hidden_state[] = {
      0.58013272, 0.58013278,  // batch1
      0.50054074, 0.50054148   // batch2
  };
  const float step3_expected_cell_state[] = {
      0.89740515, 0.8974053,  // batch1
      0.80327607, 0.80327785  // batch2
  };
  tflite::testing::TestOneStepLSTMFloat(
      step3_input_data, step3_expected_hidden_state, step3_expected_cell_state,
      hidden_state, cell_state, output);
}

TF_LITE_MICRO_TEST(TestLSTMEval) {
  std::unique_ptr<tflite::testing::ModelContents<float, float, float, float>>
      float_model_contents(
          new tflite::testing::ModelContents<float, float, float, float>(
              tflite::testing::kForgetGateParameters,
              tflite::testing::kInputGateParameters,
              tflite::testing::kCellGateParameters,
              tflite::testing::kOutputGateParameters));
  float_model_contents->SetInputTensorData(tflite::testing::kInputData);

  tflite::EvalFloatLstm(
      float_model_contents->GetTensor(0), float_model_contents->GetTensor(4),
      float_model_contents->GetTensor(1), float_model_contents->GetTensor(7),
      float_model_contents->GetTensor(10), float_model_contents->GetTensor(5),
      float_model_contents->GetTensor(2), float_model_contents->GetTensor(8),
      float_model_contents->GetTensor(11),
      /*cell_to_input_weights=*/nullptr,
      /*cell_to_forget_weights=*/nullptr,
      /*cell_to_output_weights=*/nullptr,
      /*input_layer_norm_coefficients=*/nullptr,
      /*forget_layer_norm_coefficients=*/nullptr,
      /*cell_layer_norm_coefficients=*/nullptr,
      /*output_layer_norm_coefficients=*/nullptr,
      /*aux_input=*/nullptr,
      /*aux_input_to_input_weights=*/nullptr,
      /*aux_input_to_forget_weights=*/nullptr,
      /*aux_input_to_cell_weights=*/nullptr,
      /*aux_input_to_output_weights=*/nullptr,
      float_model_contents->GetTensor(6), float_model_contents->GetTensor(3),
      float_model_contents->GetTensor(9), float_model_contents->GetTensor(12),
      /*projection_weights=*/nullptr,
      /*projection_bias=*/nullptr, &tflite::testing::kModelSettings,
      /*forward_sequence=*/true, /*time_major=*/false,
      /*output_offset=*/0, float_model_contents->ScratchBuffers(),
      float_model_contents->GetTensor(13), float_model_contents->GetTensor(14),
      float_model_contents->GetTensor(15));

  // Validate hidden state. See previous test for the calculation
  const float expected_hidden_state[] = {
      0.58013272, 0.58013278,  // batch1
      0.50054074, 0.50054148   // batch2
  };
  tflite::testing::ValidateResultGoldens(
      expected_hidden_state, float_model_contents->GetHiddenState(),
      tflite::testing::kGateOutputSize, tflite::testing::kTestFloatTolerance);
  // Validate cell state. See previous test for the calculation
  const float expected_cell_state[] = {
      0.89740515, 0.8974053,  // batch1
      0.80327607, 0.80327785  // batch2
  };
  tflite::testing::ValidateResultGoldens(
      expected_cell_state, float_model_contents->GetCellState(),
      tflite::testing::kGateOutputSize, tflite::testing::kTestFloatTolerance);

  // Validate output . See previous test for the calculation
  tflite::testing::ValidateResultGoldens(
      tflite::testing::kExpectedOutput, float_model_contents->GetOutput(),
      tflite::testing::kOutputSize, tflite::testing::kTestFloatTolerance);
}
#endif  // !defined(XTENSA)
TF_LITE_MICRO_TESTS_END
