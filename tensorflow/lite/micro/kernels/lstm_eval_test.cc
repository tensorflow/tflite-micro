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

constexpr TfLiteLSTMParams kModelSettings = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/10, /*.proj_clip=*/3,
    /*.kernel_type=*/kTfLiteLSTMFullKernel,
    /*.asymmetric_quantize_inputs=*/true};

// Test Settings
constexpr float kTestFloatTolerance = 1e-6f;

// Struct that holds the weight/bias information for a standard gate (i.e. no
// modification such as layer normalization, peephole, etc.)
struct GateParameters {
  const float activation_weight[kStateDimension * kInputDimension];
  const float recurrent_weight[kStateDimension * kStateDimension];
  const float fused_bias[kStateDimension];
};

// Parameters for different gates
// negative large weights for forget gate to make it really forget
constexpr GateParameters kForgetGateParameters = {
    /*.activation_weight=*/{-10, -10, -20, -20},
    /*.recurrent_weight=*/{-10, -10, -20, -20},
    /*.fused_bias=*/{1, 2}};
// positive large weights for input gate to make it really remember
constexpr GateParameters kInputGateParameters = {
    /*.activation_weight=*/{10, 10, 20, 20},
    /*.recurrent_weight=*/{10, 10, 20, 20},
    /*.fused_bias=*/{-1, -2}};
// all ones to test the behavior of tanh at normal range (-1,1)
constexpr GateParameters kCellGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0}};
// all ones to test the behavior of sigmoid at normal range (-1. 1)
constexpr GateParameters kOutputGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0}};

// A class that holds all the tensors for the test LSTM model
class ModelContents {
 public:
  ModelContents() {
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
    // Input/Output Tensor
    SetTensor(0, input_, input_size_);
    SetTensor(15, output_, output_size_);
  }

  TfLiteEvalTensor* GetTensor(int tensor_index) {
    return tensors_ + tensor_index;
  }

  const float* GetHiddenState() const { return hidden_state_; }
  const float* GetCellState() const { return cell_state_; }
  const float* GetOutput() const { return output_; }
  const float* GetExpectedOutput() const { return expected_output_; }

  float* ScratchBuffers() { return scratch_buffers_; }

 private:
  const GateParameters forget_gate_params_ = kForgetGateParameters;
  const GateParameters input_gate_params_ = kInputGateParameters;
  const GateParameters cell_gate_params_ = kCellGateParameters;
  const GateParameters output_gate_params_ = kOutputGateParameters;

  // states are initialized to zero
  float hidden_state_[kGateOutputSize] = {0};
  float cell_state_[kGateOutputSize] = {0};

  // Input and output
  // batch one using repeated data; batch two mimics
  // random input
  const float input_[kInputSize] = {
      0.2,   0.3,  0.2,  0.3,  0.2,  0.3,   // batch one
      -0.98, 0.62, 0.01, 0.99, 0.49, -0.32  // batch two
  };

  float output_[kOutputSize] = {0};
  // The expected model output after kTimeSteps using the fixed input and
  // parameters
  const float expected_output_[kOutputSize] = {
      0.26455893,      0.26870455,      0.47935803,
      0.47937014,      0.58013272,      0.58013278,  // batch1
      -1.41184672e-3f, -1.43329117e-5f, 0.46887168,
      0.46891281,      0.50054074,      0.50054148  // batch2
  };

  // scratch buffers (4)
  float scratch_buffers_[4 * kGateOutputSize] = {0};

  // Not const since IntArrayFromInts takes int *; the first element of the
  // array must be the size of the array
  int input_size_[4] = {3, kBatchSize, kTimeSteps, kInputDimension};
  int output_size_[4] = {3, kBatchSize, kTimeSteps, kStateDimension};
  int activation_weight_size_[3] = {2, kStateDimension, kInputDimension};
  int recurrent_weight_size_[3] = {2, kStateDimension, kStateDimension};
  int bias_size_[3] = {2, kBatchSize, kStateDimension};

  // 0 input; 1-12 gate parameters; 13-14 states; 15 output
  TfLiteEvalTensor tensors_[16];

  template <typename T>
  void SetTensor(const int index, const T* data, int* dims) {
    tensors_[index].data.data = const_cast<T*>(data);
    tensors_[index].dims = IntArrayFromInts(dims);
    tensors_[index].type = typeToTfLiteType<T>();
  }
};

template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

void TestGateOutputFloat(const GateParameters& gate_params,
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

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(CheckGateOutputFloat) {
  std::unique_ptr<tflite::testing::GateOutputCheckData> gate_output_data(
      new tflite::testing::GateOutputCheckData);

  tflite::testing::TestGateOutputFloat(
      tflite::testing::kForgetGateParameters, kTfLiteActSigmoid,
      gate_output_data->input_data, gate_output_data->hidden_state,
      gate_output_data->expected_forget_gate_output);

  tflite::testing::TestGateOutputFloat(
      tflite::testing::kInputGateParameters, kTfLiteActSigmoid,
      gate_output_data->input_data, gate_output_data->hidden_state,
      gate_output_data->expected_input_gate_output);

  tflite::testing::TestGateOutputFloat(
      tflite::testing::kOutputGateParameters, kTfLiteActSigmoid,
      gate_output_data->input_data, gate_output_data->hidden_state,
      gate_output_data->expected_output_gate_output);

  tflite::testing::TestGateOutputFloat(
      tflite::testing::kCellGateParameters,
      tflite::testing::kModelSettings.activation, gate_output_data->input_data,
      gate_output_data->hidden_state,
      gate_output_data->expected_cell_gate_output);
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
  std::unique_ptr<tflite::testing::ModelContents> model_contents(
      new tflite::testing::ModelContents);

  tflite::EvalFloatLstm(
      model_contents->GetTensor(0), model_contents->GetTensor(4),
      model_contents->GetTensor(1), model_contents->GetTensor(7),
      model_contents->GetTensor(10), model_contents->GetTensor(5),
      model_contents->GetTensor(2), model_contents->GetTensor(8),
      model_contents->GetTensor(11),
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
      /*aux_input_to_output_weights=*/nullptr, model_contents->GetTensor(6),
      model_contents->GetTensor(3), model_contents->GetTensor(9),
      model_contents->GetTensor(12),
      /*projection_weights=*/nullptr,
      /*projection_bias=*/nullptr, &tflite::testing::kModelSettings,
      /*forward_sequence=*/true, /*time_major=*/false,
      /*output_offset=*/0, model_contents->ScratchBuffers(),
      model_contents->GetTensor(13), model_contents->GetTensor(14),
      model_contents->GetTensor(15));

  // Validate hidden state. See previous test for the calculation
  const float expected_hidden_state[] = {
      0.58013272, 0.58013278,  // batch1
      0.50054074, 0.50054148   // batch2
  };
  tflite::testing::ValidateResultGoldens(
      expected_hidden_state, model_contents->GetHiddenState(),
      tflite::testing::kGateOutputSize, tflite::testing::kTestFloatTolerance);
  // Validate cell state. See previous test for the calculation
  const float expected_cell_state[] = {
      0.89740515, 0.8974053,  // batch1
      0.80327607, 0.80327785  // batch2
  };
  tflite::testing::ValidateResultGoldens(
      expected_cell_state, model_contents->GetCellState(),
      tflite::testing::kGateOutputSize, tflite::testing::kTestFloatTolerance);

  // Validate output . See previous test for the calculation
  tflite::testing::ValidateResultGoldens(
      model_contents->GetExpectedOutput(), model_contents->GetOutput(),
      tflite::testing::kOutputSize, tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TESTS_END
