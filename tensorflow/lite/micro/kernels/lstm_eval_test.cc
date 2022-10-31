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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Base class that holds input parameters for quantized and hybrid lstm.
class BaseLstmParam {
 public:
  BaseLstmParam()
      : input_size_{2, n_batch_, n_input_},
        i2i_{
            18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
            1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  0,   //
            8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6, 1, 2, 3, -4, 5,  6,   //
            1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6, 1, 7, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
            1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 8,  5,  -6,  //
            8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
            1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6, 1, 2, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  6, 1, 2, 3, 14, 5,  6,   //
            1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
        },
        i2i_size_{2, n_cell_, n_input_},
        i2f_{
            1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  0,   //
            8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1,  2, 3, -4, 5,  6,   //
            1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1,  7, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
            1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  11, 2, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  -6, 1,  2, 3, 14, 5,  6,   //
            1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
            18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
            8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  13, 2, 3, 4,  5,  6,   //
            1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 8,  5,  -6,  //
        },
        i2f_size_{2, n_cell_, n_input_},
        i2c_{
            1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  0,   //
            1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  1, 2, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  16, 1, 2, 3, 14, 5,  6,   //
            1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  7, 2, 3, 4,  5,  6,   //
            18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
            8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
            1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 8,  5,  -6,  //
            8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1, 2, 3, -4, 5,  6,   //
            1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1, 7, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
        },
        i2c_size_{2, n_cell_, n_input_},
        i2o_{
            1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1,  7, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  -1, 2, 3, 4,  5,  6,   //
            1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  1,  2, 3, 4,  -5, 6,   //
            8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  6,  1,  2, 3, 14, 5,  6,   //
            18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  -6, 1,  2, 3, 4,  5,  6,   //
            8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
            1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  0,   //
            8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1,  2, 3, -4, 5,  6,   //
            1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  -1, 2, 3, 4,  5,  6,   //
            1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 8,  5,  -6,  //
        },
        i2o_size_{2, n_cell_, n_input_},
        r2i_{
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
        },
        r2i_size_{2, n_cell_, n_output_},
        r2f_{
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
        },
        r2f_size_{2, n_cell_, n_output_},
        r2c_{
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
        },
        r2c_size_{2, n_cell_, n_output_},
        r2o_{
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
        },
        r2o_size_{2, n_cell_, n_output_},
        projection_{
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
            8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
            6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
            1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
        },
        projection_size_{2, n_cell_, n_output_},
        layer_norm_input_size_{1, n_cell_},
        layer_norm_forget_size_{1, n_cell_},
        layer_norm_cell_size_{1, n_cell_},
        layer_norm_output_size_{1, n_cell_},
        input_gate_bias_size_{1, n_cell_},
        forget_gate_bias_size_{1, n_cell_},
        cell_gate_bias_size_{1, n_cell_},
        output_gate_bias_size_{1, n_cell_},
        projection_bias_{
            16, 4, 5, 6, 1, 1  //
        },
        projection_bias_size_{1, n_output_},
        activation_size_{2, n_batch_, n_output_},
        cell_size_{2, n_batch_, n_cell_},
        output_size_{2, n_batch_, n_output_} {}

  TfLiteEvalTensor* Geti2i() {
    AssignDimsToEvalTensor(&i2i_tensor_, i2i_, i2i_size_);
    i2i_tensor_.data.int8 = i2i_;
    return &i2i_tensor_;
  }
  TfLiteEvalTensor* Geti2f() {
    AssignDimsToEvalTensor(&i2f_tensor_, i2f_, i2f_size_);
    i2f_tensor_.data.int8 = i2f_;
    return &i2f_tensor_;
  }
  TfLiteEvalTensor* Geti2c() {
    AssignDimsToEvalTensor(&i2c_tensor_, i2c_, i2c_size_);
    i2c_tensor_.data.int8 = i2c_;
    return &i2c_tensor_;
  }
  TfLiteEvalTensor* Geti2o() {
    AssignDimsToEvalTensor(&i2o_tensor_, i2o_, i2o_size_);
    i2o_tensor_.data.int8 = i2o_;
    return &i2o_tensor_;
  }
  TfLiteEvalTensor* Getr2i() {
    AssignDimsToEvalTensor(&r2i_tensor_, r2i_, r2i_size_);
    r2i_tensor_.data.int8 = r2i_;
    return &r2i_tensor_;
  }
  TfLiteEvalTensor* Getr2f() {
    AssignDimsToEvalTensor(&r2f_tensor_, r2f_, r2f_size_);
    r2f_tensor_.data.int8 = r2f_;
    return &r2f_tensor_;
  }
  TfLiteEvalTensor* Getr2c() {
    AssignDimsToEvalTensor(&r2c_tensor_, r2c_, r2c_size_);
    r2c_tensor_.data.int8 = r2c_;
    return &r2c_tensor_;
  }
  TfLiteEvalTensor* Getr2o() {
    AssignDimsToEvalTensor(&r2o_tensor_, r2o_, r2o_size_);
    r2o_tensor_.data.int8 = r2o_;
    return &r2o_tensor_;
  }
  TfLiteEvalTensor* GetProjection() {
    AssignDimsToEvalTensor(&projection_tensor_, projection_, projection_size_);
    projection_tensor_.data.int8 = projection_;
    return &projection_tensor_;
  }
  ~BaseLstmParam() {}

 protected:
  // Dimensions. Need proper size to trigger neon code.
  static const int n_batch_ = 2;
  static const int n_input_ = 18;
  static const int n_cell_ = 10;
  static const int n_output_ = 6;

  template <typename T>
  void AssignDimsToEvalTensor(TfLiteEvalTensor* tensor, T* data, int dims[]) {
    tensor->dims = IntArrayFromInts(dims);
  }

  int input_size_[3];
  TfLiteEvalTensor input_tensor_;

  // input_to_input_weights.
  int8_t i2i_[n_cell_ * n_input_];
  int i2i_size_[3];
  TfLiteEvalTensor i2i_tensor_;

  // input_to_forget_weights.
  int8_t i2f_[n_cell_ * n_input_];
  int i2f_size_[3];
  TfLiteEvalTensor i2f_tensor_;

  // input_to_cell_weights.
  int8_t i2c_[n_cell_ * n_input_];
  int i2c_size_[3];
  TfLiteEvalTensor i2c_tensor_;

  // input_to_output_weights.
  int8_t i2o_[n_cell_ * n_input_];
  int i2o_size_[3];
  TfLiteEvalTensor i2o_tensor_;

  // recurrent_to_input_weights.
  int8_t r2i_[n_cell_ * n_output_];
  int r2i_size_[3];
  TfLiteEvalTensor r2i_tensor_;

  // recurrent_to_forget_weights.
  int8_t r2f_[n_cell_ * n_output_];
  int r2f_size_[3];
  TfLiteEvalTensor r2f_tensor_;

  // recurrent_to_cell_weights.
  int8_t r2c_[n_cell_ * n_output_];
  int r2c_size_[3];
  TfLiteEvalTensor r2c_tensor_;

  // recurrent_to_output_weights.
  int8_t r2o_[n_cell_ * n_output_];
  int r2o_size_[3];
  TfLiteEvalTensor r2o_tensor_;

  // projection_weights.
  int8_t projection_[n_cell_ * n_output_];
  int projection_size_[3];
  TfLiteEvalTensor projection_tensor_;

  int layer_norm_input_size_[2];
  TfLiteEvalTensor layer_norm_input_tensor_;

  TfLiteEvalTensor layer_norm_forget_tensor_;
  int layer_norm_forget_size_[2];

  int layer_norm_cell_size_[2];
  TfLiteEvalTensor layer_norm_cell_tensor_;

  int layer_norm_output_size_[2];
  TfLiteEvalTensor layer_norm_output_tensor_;

  int input_gate_bias_size_[2];
  TfLiteEvalTensor input_gate_bias_tensor_;

  int forget_gate_bias_size_[2];
  TfLiteEvalTensor forget_gate_bias_tensor_;

  int cell_gate_bias_size_[2];
  TfLiteEvalTensor cell_gate_bias_tensor_;

  int output_gate_bias_size_[2];
  TfLiteEvalTensor output_gate_bias_tensor_;

  // projection_bias.
  int32_t projection_bias_[n_output_];

  int projection_bias_size_[2];
  TfLiteEvalTensor projection_bias_tensor_;

  int activation_size_[3];
  TfLiteEvalTensor activation_tensor_;

  int cell_size_[3];
  TfLiteEvalTensor cell_tensor_;

  int output_size_[3];
  TfLiteEvalTensor output_tensor_;
};

class QuantizedLstmParam : public BaseLstmParam {
 public:
  QuantizedLstmParam() :
    input_{
      8, 2, 3,  4, 5, 6, 1, -2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,  //
      1, 2, -3, 4, 5, 6, 1, 2,  3, 4, 5, 6, 1, 2, 3, 4, 5, 6,  //
    },
    layer_norm_input_{
      8, 2, 3, 4, 5, 6, 1, 2, 3, 4
    },
    layer_norm_forget_{
      1, 2, 3, 4, 7, 3, 4, -5, 6, 3,  //
    },
    layer_norm_cell_{
      6, 4, 5, 6, 1, 2, 3, 4, -5, 6,  //
    },
    layer_norm_output_{
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
    },
    input_gate_bias_{
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
    },
    forget_gate_bias_{
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
    },
    cell_gate_bias_{
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
    },
    output_gate_bias_{
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
    },
    activation_{
      0, 0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0, 0,  //
    },
    cell_{
      16, 4,  5, 6, 1, 1, 3, 4, -5, 6,  //
      1,  14, 5, 6, 1, 1, 3, 4, -5, 6,  //
    },
    output_{
      1, 1, 3, 4, -5, 6,  //
      1, 4, 3, 4, -5, 6,  //
    }
  {}

  // Getter methods.
  TfLiteEvalTensor* GetInput() {
    AssignDimsToEvalTensor(&input_tensor_, input_, input_size_);
    input_tensor_.data.int8 = input_;
    return &input_tensor_;
  }
  TfLiteEvalTensor* GetInputLayerNorm() {
    AssignDimsToEvalTensor(&layer_norm_input_tensor_, layer_norm_input_,
                           layer_norm_input_size_);
    layer_norm_input_tensor_.data.i16 = layer_norm_input_;
    return &layer_norm_input_tensor_;
  }
  TfLiteEvalTensor* GetForgetLayerNorm() {
    AssignDimsToEvalTensor(&layer_norm_forget_tensor_, layer_norm_forget_,
                           layer_norm_forget_size_);
    layer_norm_forget_tensor_.data.i16 = layer_norm_forget_;
    return &layer_norm_forget_tensor_;
  }
  TfLiteEvalTensor* GetCellLayerNorm() {
    AssignDimsToEvalTensor(&layer_norm_cell_tensor_, layer_norm_cell_,
                           layer_norm_cell_size_);
    layer_norm_cell_tensor_.data.i16 = layer_norm_cell_;
    return &layer_norm_cell_tensor_;
  }
  TfLiteEvalTensor* GetOutputLayerNorm() {
    AssignDimsToEvalTensor(&layer_norm_output_tensor_, layer_norm_output_,
                           layer_norm_output_size_);
    layer_norm_output_tensor_.data.i16 = layer_norm_output_;
    return &layer_norm_output_tensor_;
  }
  TfLiteEvalTensor* GetInputBias() {
    AssignDimsToEvalTensor(&input_gate_bias_tensor_, input_gate_bias_,
                           input_gate_bias_size_);
    input_gate_bias_tensor_.data.i32 = input_gate_bias_;
    return &input_gate_bias_tensor_;
  }
  TfLiteEvalTensor* GetForgetBias() {
    AssignDimsToEvalTensor(&forget_gate_bias_tensor_, forget_gate_bias_,
                           forget_gate_bias_size_);
    forget_gate_bias_tensor_.data.i32 = forget_gate_bias_;
    return &forget_gate_bias_tensor_;
  }
  TfLiteEvalTensor* GetCellBias() {
    AssignDimsToEvalTensor(&cell_gate_bias_tensor_, cell_gate_bias_,
                           cell_gate_bias_size_);
    cell_gate_bias_tensor_.data.i32 = cell_gate_bias_;
    return &cell_gate_bias_tensor_;
  }
  TfLiteEvalTensor* GetOutputBias() {
    AssignDimsToEvalTensor(&output_gate_bias_tensor_, output_gate_bias_,
                           output_gate_bias_size_);
    output_gate_bias_tensor_.data.i32 = output_gate_bias_;
    return &output_gate_bias_tensor_;
  }
  TfLiteEvalTensor* GetProjectionBias() {
    AssignDimsToEvalTensor(&projection_bias_tensor_, projection_bias_,
                           projection_bias_size_);
    projection_bias_tensor_.data.i32 = projection_bias_;
    return &projection_bias_tensor_;
  }

  // Set up quantization parameters.
  tflite::IntegerLstmParameter* GetQuantParam() {
    integer_lstm_param_.effective_input_to_input_scale_a = 1808677632;
    integer_lstm_param_.effective_input_to_input_scale_b = -1;
    integer_lstm_param_.effective_recurrent_to_input_scale_a = 1078887680;
    integer_lstm_param_.effective_recurrent_to_input_scale_b = -1;
    integer_lstm_param_.effective_cell_to_input_scale_a = 1073741824;
    integer_lstm_param_.effective_cell_to_input_scale_b = 1;
    integer_lstm_param_.effective_input_to_forget_scale_a = 1845996800;
    integer_lstm_param_.effective_input_to_forget_scale_b = -3;
    integer_lstm_param_.effective_recurrent_to_forget_scale_a = 1477412736;
    integer_lstm_param_.effective_recurrent_to_forget_scale_b = -2;
    integer_lstm_param_.effective_cell_to_forget_scale_a = 1073741824;
    integer_lstm_param_.effective_cell_to_forget_scale_b = 1;
    integer_lstm_param_.effective_input_to_cell_scale_a = 1648385408;
    integer_lstm_param_.effective_input_to_cell_scale_b = -2;
    integer_lstm_param_.effective_recurrent_to_cell_scale_a = 1185544192,
    integer_lstm_param_.effective_recurrent_to_cell_scale_b = -1;
    integer_lstm_param_.effective_input_to_output_scale_a = 1328153600;
    integer_lstm_param_.effective_input_to_output_scale_b = -1;
    integer_lstm_param_.effective_recurrent_to_output_scale_a = 1479582592;
    integer_lstm_param_.effective_recurrent_to_output_scale_b = -1;
    integer_lstm_param_.effective_cell_to_output_scale_a = 1073741824,
    integer_lstm_param_.effective_cell_to_output_scale_b = 1;
    integer_lstm_param_.effective_proj_scale_a = 1105682560;
    integer_lstm_param_.effective_proj_scale_b = -8;
    integer_lstm_param_.effective_hidden_scale_a = 0;
    integer_lstm_param_.effective_hidden_scale_b = 0;
    integer_lstm_param_.layer_norm_input_scale_a = 2011617664;
    integer_lstm_param_.layer_norm_input_scale_b = -11;
    integer_lstm_param_.layer_norm_forget_scale_a = 1968024960;
    integer_lstm_param_.layer_norm_forget_scale_b = -13;
    integer_lstm_param_.layer_norm_cell_scale_a = 1097334528,
    integer_lstm_param_.layer_norm_cell_scale_b = -12;
    integer_lstm_param_.layer_norm_output_scale_a = 1837163008;
    integer_lstm_param_.layer_norm_output_scale_b = -12;
    integer_lstm_param_.quantized_cell_clip = 20480;
    integer_lstm_param_.quantized_proj_clip = 0;
    integer_lstm_param_.cell_scale = -11;
    integer_lstm_param_.input_variance_guard = 1;
    integer_lstm_param_.forget_variance_guard = 2;
    integer_lstm_param_.cell_variance_guard = 2;
    integer_lstm_param_.output_variance_guard = 1;
    integer_lstm_param_.hidden_zp = 0;
    integer_lstm_param_.input_to_forget_effective_bias =
        input_to_forget_effective_bias;
    integer_lstm_param_.recurrent_to_forget_effective_bias =
        recurrent_to_forget_effective_bias;
    integer_lstm_param_.input_to_cell_effective_bias =
        input_to_cell_effective_bias;
    integer_lstm_param_.recurrent_to_cell_effective_bias =
        recurrent_to_cell_effective_bias;
    integer_lstm_param_.input_to_output_effective_bias =
        input_to_output_effective_bias;
    integer_lstm_param_.recurrent_to_output_effective_bias =
        recurrent_to_output_effective_bias;
    integer_lstm_param_.input_to_input_effective_bias =
        input_to_input_effective_bias;
    integer_lstm_param_.recurrent_to_input_effective_bias =
        recurrent_to_input_effective_bias;
    integer_lstm_param_.projection_effective_bias = projection_effective_bias;
    for (int i = 0; i < n_cell_; ++i) {
      input_to_forget_effective_bias[i] = 152;
      recurrent_to_forget_effective_bias[i] = 315;
      input_to_cell_effective_bias[i] = 165;
      recurrent_to_cell_effective_bias[i] = 1165;
      input_to_output_effective_bias[i] = 159;
      recurrent_to_output_effective_bias[i] = 915;
      input_to_input_effective_bias[i] = -15;
      recurrent_to_input_effective_bias[i] = 315;
    }
    for (int i = 0; i < n_output_; ++i) {
      projection_effective_bias[i] = 115;
    }
    return &integer_lstm_param_;
  }

  // Create scratch buffers.
  int16_t* GetScratch0() { return scratch0_; }
  int16_t* GetScratch1() { return scratch1_; }
  int16_t* GetScratch2() { return scratch2_; }
  int16_t* GetScratch3() { return scratch3_; }
  int8_t* GetScratch4() { return scratch4_; }
  int32_t* GetScratch5() { return scratch5_; }
  TfLiteEvalTensor* GetActivation() {
    AssignDimsToEvalTensor(&activation_tensor_, activation_, activation_size_);
    activation_tensor_.data.int8 = activation_;
    return &activation_tensor_;
  }
  TfLiteEvalTensor* GetOutput() {
    AssignDimsToEvalTensor(&output_tensor_, output_, output_size_);
    output_tensor_.data.int8 = output_;
    return &output_tensor_;
  }
  TfLiteEvalTensor* GetCell() {
    AssignDimsToEvalTensor(&cell_tensor_, cell_, cell_size_);
    cell_tensor_.data.i16 = cell_;
    return &cell_tensor_;
  }
  ~QuantizedLstmParam() {}

 private:
  // input.
  int8_t input_[n_batch_ * n_input_];

  int16_t layer_norm_input_[n_cell_];

  // forget_layer_norm_coefficient.
  int16_t layer_norm_forget_[n_cell_];

  // cell_layer_norm_coefficients.
  int16_t layer_norm_cell_[n_cell_];

  // output_layer_norm_coefficients.
  int16_t layer_norm_output_[n_cell_];

  // input_gate_bias.
  int32_t input_gate_bias_[n_cell_];

  // forget_gate_bias.
  int32_t forget_gate_bias_[n_cell_];

  // cell_gate_bias.
  int32_t cell_gate_bias_[n_cell_];

  // output_gate_bias.
  int32_t output_gate_bias_[n_cell_];

  // activation.
  int8_t activation_[n_batch_ * n_output_];

  // cell.
  int16_t cell_[n_batch_ * n_cell_];

  // output.
  int8_t output_[n_batch_ * n_output_];

  // quantized_lstm_param
  tflite::IntegerLstmParameter integer_lstm_param_;

  int32_t input_to_forget_effective_bias[n_cell_];
  int32_t recurrent_to_forget_effective_bias[n_cell_];
  int32_t input_to_cell_effective_bias[n_cell_];
  int32_t recurrent_to_cell_effective_bias[n_cell_];
  int32_t input_to_output_effective_bias[n_cell_];
  int32_t recurrent_to_output_effective_bias[n_cell_];
  int32_t input_to_input_effective_bias[n_cell_];
  int32_t recurrent_to_input_effective_bias[n_cell_];
  int32_t projection_effective_bias[n_output_];

  int16_t scratch0_[n_batch_ * n_cell_];
  int16_t scratch1_[n_batch_ * n_cell_];
  int16_t scratch2_[n_batch_ * n_cell_];
  int16_t scratch3_[n_batch_ * n_cell_];
  int8_t scratch4_[n_batch_ * n_cell_];
  int32_t scratch5_[n_batch_ * n_cell_];
};

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)

// Ensures that a regular set and get pair works ok.
TF_LITE_MICRO_TEST(TestOneFullyQuantizedLSTM) {
  tflite::testing::QuantizedLstmParam one_parameter;
  auto activation = one_parameter.GetActivation();
  auto output = one_parameter.GetOutput();
  auto cell = one_parameter.GetCell();
  auto param = one_parameter.GetQuantParam();
  tflite::EvalInteger8x8_16Lstm(
      one_parameter.GetInput(), one_parameter.Geti2i(), one_parameter.Geti2f(),
      one_parameter.Geti2c(), one_parameter.Geti2o(), one_parameter.Getr2i(),
      one_parameter.Getr2f(), one_parameter.Getr2c(), one_parameter.Getr2o(),
      nullptr, nullptr, nullptr, one_parameter.GetInputLayerNorm(),
      one_parameter.GetForgetLayerNorm(), one_parameter.GetCellLayerNorm(),
      one_parameter.GetOutputLayerNorm(), one_parameter.GetInputBias(),
      one_parameter.GetForgetBias(), one_parameter.GetCellBias(),
      one_parameter.GetOutputBias(), one_parameter.GetProjection(),
      one_parameter.GetProjectionBias(), nullptr, /*forward_sequence=*/true,
      /*time_major=*/true, param, 50, activation, cell, output,
      one_parameter.GetScratch0(), one_parameter.GetScratch1(),
      one_parameter.GetScratch2(), one_parameter.GetScratch3(),
      one_parameter.GetScratch4(), nullptr);

  // Verify results.
  const int16_t expected_cell[20] = {
      7, 2, 3, 3, 0, 1, 0, 2, -2, 3, 1, 6, 3, 3, 0, 1, 0, 2, -2, 3,
  };
  const int8_t expected_activation[12] = {
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
  };
  for (int i = 0; i < 20; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_cell[i], cell->data.i16[i]);
  }
  for (int i = 0; i < 12; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_activation[i], activation->data.int8[i]);
  }
  for (int i = 0; i < 12; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_activation[i], output->data.int8[i]);
  }
}
#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_END
