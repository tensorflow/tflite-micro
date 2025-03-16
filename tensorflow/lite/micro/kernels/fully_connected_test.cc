/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Simple test data for 2x2x10 input 2x3x10 weights.
const int simple_input_size = 20;
int simple_input_dims[] = {2, 2, 10};
const float simple_input_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
    1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
};
const int simple_weights_size = 30;
int simple_weights_dims[] = {2, 3, 10};
const float simple_weights_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
};

int simple_bias_dims[] = {1, 3};
const float simple_bias_data[] = {1, 2, 3};

#if (defined(USE_TFLM_COMPRESSION) || (!defined(XTENSA) && !defined(HEXAGON)))

constexpr size_t simple_bias_size =
    std::extent<decltype(simple_bias_data)>::value;

#endif  // (defined(USE_TFLM_COMPRESSION) || (!defined(XTENSA) &&
        // !defined(HEXAGON)))

#ifdef USE_TFLM_COMPRESSION

// compressed filter data for kBinQuant scheme
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantWeightData[] = {
    0x01, 0x23, 0x45, 0x67, 0x89, 0x01, 0x23, 0x45,
    0x67, 0x89, 0x01, 0x23, 0x45, 0x67, 0x89};
constexpr float kBinQuantWeightValueTable[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
constexpr size_t kBinQuantWeightValueTableElements =
    std::extent<decltype(tflite::testing::kBinQuantWeightValueTable)>::value;
constexpr int kBinQuantWeightBitWidth = 4;
// compressed bias data for kBinQuant scheme
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantBiasData[] = {0x18};
constexpr int kBinQuantBiasBitWidth = 2;

#endif  // USE_TFLM_COMPRESSION

// TODO(b/258710417): INT4 isn't currently supported on Hexagon.
#if !defined(HEXAGON)
const float simple_int4_weights_data[] = {
    -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,  // u = 0
    -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,  // u = 1
    -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,  // u = 2
};
const float simple_golden_null_bias_int4_weights[] = {
    -28, -28, -28, 0, 0, 0,
};
#endif
const float simple_golden[] = {
    24, 25, 26, 58, 59, 60,
};
const float simple_golden_null_bias[] = {
    23, 23, 23, 57, 57, 57,
};

const int simple_output_size = 6;
int simple_output_dims[] = {2, 2, 3};

// Test data for 2x2x10 input 2x3x10 weights with negative outputs to test relu.
const int relu_input_size = 20;
int relu_input_dims[] = {2, 2, 10};
const float relu_input_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
    1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
};
const int relu_weights_size = 30;
int relu_weights_dims[] = {2, 3, 10};
const float relu_weights_data[] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 0
    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,  // u = 1
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 2
};
int relu_bias_dims[] = {1, 3};
const float relu_bias_data[] = {1, -2, 3};
const float relu_golden[] = {
    24, 0, 26, 58, 0, 60,
};
const int relu_output_size = 6;
int relu_output_dims[] = {2, 2, 3};

// Input and filter similar to real model. Input shape is 1x64 and output is
// 1x16.
const int representative_64x16_input_size = 64;
int representative_64x16_input_dims[] = {2, 1, 64};
const float representative_64x16_input_data[] = {
    0.0000, 0.1543, 0.0000, 0.0000, 1.8520, 0.0000, 4.7844, 1.1832,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.5948, 0.0000,
    1.5948, 1.9549, 0.0000, 1.2347, 0.0000, 1.5948, 1.5948, 0.5145,
    0.0000, 0.0000, 0.0000, 0.0000, 2.6237, 0.0000, 0.0000, 0.0000,
    1.3890, 5.3503, 2.3665, 2.9838, 0.0000, 1.2861, 0.0000, 3.0867,
    0.9775, 0.0000, 5.9676, 0.0000, 0.0000, 1.4405, 0.5145, 2.5723,
    3.1896, 4.4757, 0.0000, 0.0000, 0.0000, 0.0000, 4.1671, 0.0000,
    2.8295, 3.0353, 0.0000, 2.7780, 0.0000, 0.0000, 0.0000, 0.0000};
const int representative_64x16_weights_size = 64 * 16;
int representative_64x16_weights_dims[] = {2, 16, 64};
const float representative_64x16_weights_data[] = {
    -0.1075, 0.1245,  0.1811,  -0.1302, -0.1868, 0.0679,  0.1245,  0.2321,
    -0.1981, -0.2094, 0.1358,  -0.1698, 0.0113,  0.0566,  0.1358,  -0.2490,
    0.0000,  -0.1189, -0.0170, -0.0396, -0.3113, 0.1641,  -0.4188, 0.0566,
    -0.4471, 0.4754,  -0.0396, 0.0113,  -0.0340, 0.0170,  0.0170,  0.1811,
    -0.0792, 0.4981,  0.2490,  -0.1924, 0.0792,  0.1868,  -0.1075, -0.3962,
    0.1358,  0.2547,  -0.1245, -0.0962, -0.0283, 0.4132,  -0.0057, -0.5150,
    0.1019,  0.1585,  -0.0962, -0.2207, -0.2377, 0.2830,  0.4471,  0.0170,
    0.0566,  0.2038,  0.1019,  -0.0226, 0.2830,  0.1415,  0.0283,  -0.0792,
    0.4301,  0.3226,  -0.1132, 0.4981,  -0.3849, -0.2943, -0.2547, -0.2264,
    0.0453,  -0.0170, 0.0396,  0.1415,  0.3000,  0.2547,  0.0962,  0.2151,
    -0.1585, -0.1302, -0.0057, -0.2773, 0.0283,  -0.0906, 0.1302,  -0.1075,
    -0.0566, 0.1755,  0.2773,  0.0283,  0.0566,  0.1528,  -0.0736, -0.2830,
    0.0792,  0.0962,  -0.2321, -0.0113, 0.2660,  -0.2887, -0.0566, 0.0057,
    -0.2547, -0.0679, -0.2321, 0.0340,  0.1868,  0.2490,  0.2264,  -0.3509,
    0.1585,  -0.0849, -0.0623, 0.1132,  0.3396,  -0.2490, 0.1528,  0.0679,
    0.1755,  0.4754,  -0.0057, -0.2151, -0.1415, -0.1302, -0.2717, 0.1641,
    0.5037,  -0.2321, 0.0170,  -0.1755, -0.1075, -0.0226, 0.2038,  -0.0340,
    -0.5150, -0.3113, 0.1472,  -0.0226, 0.1528,  0.1189,  -0.1472, 0.0396,
    -0.3000, -0.1924, -0.0283, 0.0283,  0.1641,  0.0736,  0.1472,  -0.1755,
    -0.1132, 0.0113,  -0.1868, -0.2604, -0.3283, -0.0509, 0.0283,  -0.0679,
    0.0623,  0.0792,  -0.0283, -0.0962, 0.0396,  0.1641,  0.4584,  0.3226,
    0.0226,  -0.1811, 0.2377,  -0.1019, 0.2321,  0.1811,  -0.1924, -0.0057,
    0.0736,  0.0113,  0.2547,  -0.2264, -0.0170, -0.0396, 0.1245,  -0.1415,
    0.1755,  0.3679,  -0.2377, -0.0396, -0.1585, -0.3000, -0.1641, -0.1302,
    -0.0396, -0.1698, 0.1189,  0.2434,  0.1132,  -0.1245, -0.1415, 0.0453,
    0.1868,  -0.0906, -0.1189, -0.0509, 0.0057,  -0.1189, -0.0057, 0.0170,
    -0.1924, 0.2207,  0.0792,  -0.4641, -0.2660, 0.2943,  0.1358,  -0.0340,
    -0.3339, -0.1189, 0.0906,  -0.4358, 0.0453,  -0.1755, 0.1415,  0.0340,
    0.1924,  -0.0057, 0.2321,  -0.2094, -0.1132, 0.0000,  0.1924,  -0.3000,
    0.0340,  -0.3396, -0.0906, -0.0340, 0.1641,  -0.0226, -0.1472, -0.1019,
    0.2377,  -0.0962, -0.3396, -0.5433, 0.0906,  0.2151,  -0.0679, 0.1755,
    0.1528,  0.0283,  -0.4188, -0.0340, -0.0057, -0.0679, 0.0509,  0.1472,
    -0.3849, -0.0113, 0.3962,  0.0849,  0.1472,  0.0340,  -0.1358, 0.1641,
    -0.2038, 0.2151,  -0.1189, -0.3679, 0.0906,  -0.0679, 0.5716,  -0.0057,
    -0.0736, 0.0113,  0.2830,  -0.2887, 0.0396,  0.0849,  -0.0736, -0.0736,
    -0.3679, 0.2264,  0.0113,  -0.1641, 0.0396,  -0.1132, -0.0623, 0.3113,
    0.5999,  -0.1415, 0.1472,  -0.2038, -0.1132, -0.2377, 0.0566,  0.1755,
    -0.0057, -0.0453, 0.0226,  0.1132,  0.1698,  0.0340,  -0.0226, 0.0226,
    0.4415,  -0.3792, 0.0792,  0.3736,  -0.5999, -0.3056, -0.1924, -0.1132,
    -0.0962, 0.0283,  0.0000,  -0.3339, -0.3226, 0.3679,  -0.0453, -0.1641,
    0.0170,  0.1302,  -0.0170, -0.0509, 0.1755,  -0.0283, -0.1302, -0.2887,
    -0.0679, 0.0340,  0.4641,  0.2321,  0.7188,  0.3339,  -0.1075, 0.4754,
    -0.0226, 0.3226,  -0.1528, -0.0849, 0.0509,  -0.1981, 0.0113,  0.2321,
    0.2773,  -0.1019, 0.4075,  0.0396,  0.0792,  0.1132,  -0.0906, -0.4188,
    0.1924,  -0.3679, -0.6396, 0.1358,  0.4981,  0.4132,  -0.0283, 0.3849,
    -0.3509, -0.0566, -0.0962, 0.3113,  -0.1811, 0.4019,  0.0453,  -0.0057,
    -0.1868, -0.2490, -0.0792, -0.3622, 0.1924,  -0.0453, -0.1528, -0.1811,
    0.5943,  -0.1302, 0.3170,  -0.0170, 0.0509,  -0.1528, -0.1755, 0.5547,
    0.2490,  -0.0906, 0.0000,  0.1698,  0.0000,  0.0340,  -0.1132, -0.0509,
    -0.1755, -0.2943, 0.1472,  0.0849,  0.0000,  0.1528,  -0.0566, 0.1528,
    -0.5264, -0.5320, -0.0736, 0.0566,  0.2604,  -0.4075, 0.0962,  -0.3453,
    -0.1415, 0.0057,  0.3905,  0.2830,  0.3679,  0.5320,  -0.2660, 0.0340,
    0.0736,  0.0057,  0.2207,  0.4471,  0.0849,  0.3000,  -0.0057, -0.0623,
    0.1415,  -0.0566, 0.5264,  -0.0340, 0.0226,  -0.0623, -0.0113, -0.5037,
    -0.4471, 0.0170,  -0.0396, -0.1358, -0.1698, 0.1924,  0.0057,  -0.1585,
    0.0849,  -0.1698, 0.0057,  -0.1245, -0.0170, -0.1755, -0.0792, 0.5264,
    0.1358,  0.2434,  0.1585,  -0.4188, -0.1472, -0.1358, -0.0849, -0.1189,
    0.5037,  0.0736,  -0.0453, -0.2434, 0.1868,  -0.0679, 0.1415,  -0.2717,
    0.2604,  0.0057,  -0.1528, -0.1811, 0.0226,  -0.1641, 0.3170,  -0.1981,
    0.1245,  0.0226,  0.0566,  0.2830,  -0.1755, 0.0396,  -0.2094, 0.1924,
    0.1698,  0.0283,  0.1641,  0.0849,  0.0000,  -0.1698, -0.1415, -0.3000,
    0.4471,  0.3056,  -0.0283, -0.4245, -0.0453, 0.0226,  0.0000,  -0.1075,
    -0.1528, -0.3226, 0.2773,  -0.2264, -0.1811, 0.1755,  -0.3566, -0.4188,
    0.1755,  -0.0057, 0.2038,  0.1075,  0.3679,  -0.0792, 0.2207,  -0.0453,
    0.3736,  0.2943,  -0.0113, -0.0623, 0.2264,  0.0113,  -0.0396, -0.2207,
    0.0453,  -0.2830, -0.1302, 0.0623,  -0.1924, -0.1811, -0.2717, 0.2830,
    0.2094,  0.0170,  -0.3170, -0.0283, -0.1189, -0.0509, -0.0566, -0.3622,
    0.1132,  -0.0906, 0.1132,  0.4019,  -0.4698, -0.1019, -0.1075, -0.2094,
    -0.2207, -0.0509, 0.0057,  0.1019,  -0.0509, 0.2264,  -0.5716, 0.0226,
    -0.4019, 0.1641,  -0.3000, 0.3849,  0.1245,  0.0679,  0.3056,  0.2377,
    0.0679,  -0.0170, -0.5377, -0.0170, 0.0057,  0.1358,  -0.1132, -0.2038,
    0.0679,  0.1075,  -0.2773, 0.5943,  0.0623,  -0.1472, 0.3566,  0.0396,
    -0.2377, 0.2604,  0.0849,  0.1358,  -0.3792, -0.0340, -0.1415, 0.3566,
    -0.3736, 0.1245,  0.0566,  0.3396,  0.0736,  0.4019,  -0.1528, 0.1075,
    0.0792,  -0.2547, 0.0453,  -0.1755, 0.1868,  -0.2547, 0.1075,  0.0623,
    0.1698,  -0.0170, 0.1585,  -0.0736, -0.4358, -0.0113, -0.6792, -0.0849,
    -0.0396, -0.6056, 0.1358,  0.1189,  0.2547,  0.1528,  0.2887,  0.0453,
    -0.1075, -0.3283, -0.0453, -0.0509, 0.2038,  0.2547,  0.0849,  -0.0566,
    -0.1698, 0.0509,  -0.0113, -0.1585, 0.1924,  -0.0792, -0.1868, 0.0509,
    -0.1698, -0.0849, -0.0170, 0.0453,  0.3170,  0.0906,  -0.5943, -0.1245,
    0.1585,  -0.1755, -0.2151, 0.0906,  0.1924,  0.3170,  -0.2490, -0.5660,
    -0.0283, 0.0962,  -0.1358, 0.1585,  0.0057,  -0.2604, 0.1189,  -0.0170,
    0.3509,  0.0623,  0.0679,  -0.1302, -0.0792, 0.0906,  -0.0792, 0.0849,
    -0.1924, 0.2604,  -0.1245, -0.3679, 0.0340,  0.0113,  -0.1698, 0.2490,
    0.0283,  0.1019,  -0.3736, 0.1019,  -0.2207, -0.0340, 0.3170,  0.1755,
    0.0962,  0.3226,  -0.0113, -0.1189, -0.2321, -0.0226, -0.2434, -0.0170,
    -0.1585, -0.0283, -0.1132, 0.0679,  -0.4188, -0.0453, 0.1528,  -0.1302,
    -0.3792, 0.1415,  -0.1358, -0.1811, 0.1302,  0.1415,  0.5207,  0.0509,
    -0.1358, -0.0396, -0.2434, 0.0396,  0.0792,  -0.2264, -0.1415, 0.0906,
    0.1245,  0.0170,  0.0623,  -0.1415, 0.2773,  -0.3566, -0.0396, 0.2887,
    0.4188,  0.1698,  -0.2547, 0.1132,  -0.0453, -0.0113, -0.1358, 0.1075,
    0.0566,  0.1075,  0.2604,  -0.0849, -0.2490, 0.1415,  0.0509,  -0.2151,
    0.0340,  0.1698,  0.0509,  -0.0906, 0.0566,  -0.1075, -0.2151, 0.2038,
    -0.1924, -0.0113, 0.2830,  0.1358,  -0.1189, 0.0113,  -0.5603, -0.2830,
    -0.2943, 0.0453,  -0.0396, 0.1358,  0.0566,  0.2038,  -0.3283, -0.0509,
    0.0509,  0.1641,  0.2094,  -0.2038, -0.1868, -0.1585, -0.2207, -0.1302,
    0.0396,  -0.1019, -0.0679, 0.1075,  -0.4584, -0.2207, 0.2434,  -0.0113,
    0.0849,  0.1755,  -0.3056, 0.1585,  -0.2547, 0.0453,  0.0906,  -0.1358,
    -0.0679, -0.0509, 0.0679,  -0.3509, 0.0057,  0.0453,  0.4132,  -0.1981,
    0.2264,  -0.0736, 0.1075,  0.0679,  -0.0906, -0.3113, 0.0509,  0.0849,
    0.2604,  0.0623,  -0.3113, 0.3849,  0.0000,  0.6396,  -0.2038, -0.1019,
    0.1245,  -0.0453, 0.1641,  0.1075,  -0.1075, -0.2660, -0.4528, -0.0566,
    -0.0170, 0.0453,  0.0340,  0.1189,  -0.2434, -0.0283, -0.1811, 0.2547,
    0.0000,  -0.0226, 0.4471,  0.1019,  -0.1472, 0.0849,  0.1075,  0.1075,
    0.0283,  -0.2773, 0.4415,  -0.1811, 0.2717,  0.3170,  0.0509,  0.0623,
    -0.0962, 0.1585,  -0.0792, -0.1811, -0.0792, -0.3283, 0.0962,  -0.1698,
    -0.0736, 0.0453,  0.0962,  -0.3566, -0.4584, 0.3396,  -0.4811, 0.3056,
    -0.1755, 0.2490,  -0.1698, -0.2377, -0.3339, -0.0453, 0.1811,  0.0736,
    0.0340,  -0.0962, -0.0113, -0.3056, -0.3339, 0.2038,  0.2038,  -0.1924,
    0.2547,  -0.4471, -0.0849, -0.2038, 0.3566,  -0.4811, 0.3453,  0.0849,
    0.1189,  0.3170,  -0.1358, 0.2717,  0.0113,  -0.4754, -0.1924, 0.4245,
    -0.2773, 0.3453,  0.2264,  0.2943,  0.5320,  0.2773,  -0.2264, -0.1019,
    -0.1132, -0.3962, 0.3679,  0.0509,  -0.0623, -0.0906, -0.5603, -0.1641,
    -0.3170, -0.2377, 0.1415,  -0.0509, 0.0792,  0.0170,  -0.0226, -0.0057,
    -0.1358, -0.4245, 0.3905,  0.3113,  0.0340,  -0.1189, 0.2887,  -0.2943,
    -0.3056, 0.2434,  0.1019,  -0.0170, 0.3849,  0.1528,  -0.0736, -0.0170,
    0.0792,  0.1755,  0.0509,  0.3509,  0.1472,  0.1528,  0.1472,  0.0057,
    0.0113,  -0.0113, -0.3283, -0.3962, -0.0792, -0.1245, -0.0283, -0.1868,
    0.4019,  0.2943,  -0.0906, -0.2321, 0.6056,  0.1189,  0.0340,  -0.2207,
    -0.0453, 0.3339,  0.2377,  -0.1641, 0.3736,  0.2151,  -0.2547, 0.0453,
    0.1924,  -0.1019, -0.0340, -0.2207, 0.3962,  -0.4471, -0.2547, -0.2151,
    -0.3736, 0.0283,  0.1189,  0.0283,  0.0736,  0.0396,  0.1019,  0.0283,
    0.0170,  0.2321,  0.3509,  -0.0226, -0.0226, 0.0736,  0.0283,  0.1641,
    -0.0906, 0.1811,  0.0226,  0.5716,  -0.0396, -0.0509, -0.1641, -0.0509,
    0.4132,  -0.2604, 0.1019,  -0.0283, -0.0340, 0.0453,  0.1472,  -0.0057,
    0.2717,  -0.2094, 0.3396,  0.0340,  0.1245,  0.2547,  -0.5886, 0.2717,
    -0.0906, 0.1641,  0.0962,  -0.0792, -0.0113, 0.2264,  -0.0736, 0.3170,
    0.0623,  0.0679,  0.0623,  -0.0792, -0.2207, 0.1924,  0.1245,  -0.2773};
int representative_64x16_bias_dims[] = {1, 16};
const float representative_64x16_bias_data[] = {
    -0.0084, 0.0006,  0.0000,  0.0000,  -0.0087, -0.0006, -0.0003, -0.0003,
    0.0006,  -0.0003, -0.0003, -0.0003, -0.0253, 0.0012,  0.0000,  0.0000};
const float representative_64x16_golden[] = {
    3.8624,  -2.9580, 4.3043,  -1.2844, -1.5769, -2.7998, -0.1011, -3.4029,
    -1.0557, -7.1931, -1.4852, -0.4163, 1.7186,  -0.6965, 0.3580,  2.7378};
const int representative_64x16_output_size = 16;
int representative_64x16_output_dims[] = {2, 1, 16};

constexpr int kMaxTensors = 4;

template <typename T, typename TW = void, typename TB = void>
TfLiteStatus ValidateFullyConnectedGoldens(
    TfLiteTensor* tensors, const int tensors_size, bool null_bias,
    const TfLiteFusedActivation activation, const float tolerance,
    const int output_len, const T* golden, T* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<TW>* weight_comp_info = nullptr,
    const TestCompressionInfo<TB>* bias_comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  TfLiteFullyConnectedParams builtin_data = {
      activation, kTfLiteFullyConnectedWeightsFormatDefault, false, false,
      kTfLiteNoType};

  // Avoid variable length array warning.
  constexpr int inputs_array_len = 4;
  constexpr int outputs_array_len = 2;
  int inputs_array_data[inputs_array_len];
  int outputs_array_data[outputs_array_len];

  outputs_array_data[0] = 1;
  inputs_array_data[1] = 0;
  inputs_array_data[2] = 1;

  if (null_bias) {
    inputs_array_data[0] = 2;
    outputs_array_data[1] = 2;
  } else {
    inputs_array_data[0] = 3;
    inputs_array_data[3] = 2;
    outputs_array_data[1] = 3;
  }

  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

#ifdef USE_TFLM_COMPRESSION

  TestCompressedList<kMaxTensors> tcl;

  if (weight_comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(
        tcl.AddInput(*weight_comp_info, tensors[kFullyConnectedWeightsTensor],
                     kFullyConnectedWeightsTensor),
        kTfLiteOk);
    TF_LITE_MICRO_CHECK_FAIL();
  }
  if (bias_comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(
        tcl.AddInput(*bias_comp_info, tensors[kFullyConnectedBiasTensor],
                     kFullyConnectedBiasTensor),
        kTfLiteOk);
    TF_LITE_MICRO_CHECK_FAIL();
  }
  const CompressedTensorList* comp_list_p = tcl.GetCompressedTensorList();

#endif  // USE_TFLM_COMPRESSION

  const TFLMRegistration registration = Register_FULLY_CONNECTED();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data), nullptr
#ifdef USE_TFLM_COMPRESSION
                             ,
                             comp_list_p
#endif  // USE_TFLM_COMPRESSION
  );

  TfLiteStatus status = runner.InitAndPrepare();
  if (status != kTfLiteOk) {
    return status;
  }

  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
  return kTfLiteOk;
}

TfLiteStatus TestFullyConnectedFloat(
    int* input_dims_data, const float* input_data, int* weights_dims_data,
    const float* weights_data, int* bias_dims_data, const float* bias_data,
    const float* golden, int* output_dims_data,
    TfLiteFusedActivation activation, float* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<const float>* weight_comp_info = nullptr,
    const TestCompressionInfo<const float>* bias_comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  bool null_bias = bias_data == nullptr ? true : false;

  const int inputs_size = null_bias ? 2 : 3;
  constexpr int outputs_size = 1;
  const int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[kMaxTensors];

  tensors[0] = CreateTensor(input_data, input_dims);
  tensors[1] = CreateTensor(weights_data, weights_dims);

  if (null_bias) {
    tensors[2] = CreateTensor(output_data, output_dims);
  } else {
    tensors[2] = CreateTensor(bias_data, bias_dims);
    tensors[3] = CreateTensor(output_data, output_dims);
  }

  return ValidateFullyConnectedGoldens(tensors, tensors_size, null_bias,
                                       activation, 1e-4f, output_dims_count,
                                       golden, output_data
#ifdef USE_TFLM_COMPRESSION
                                       ,
                                       weight_comp_info, bias_comp_info
#endif  // USE_TFLM_COMPRESSION
  );
}

template <typename dataT, typename weightT, typename biasT>
TfLiteStatus TestFullyConnectedQuantized(
    int* input_dims_data, const float* input_data, dataT* input_quantized,
    const float input_scale, const int input_zero_point, int* weights_dims_data,
    const float* weights_data, weightT* weights_quantized,
    const float weights_scale, const int weights_zero_point,
    int* bias_dims_data, const float* bias_data, biasT* bias_quantized,
    const float* golden, dataT* golden_quantized, int* output_dims_data,
    const float output_scale, const int output_zero_point,
    TfLiteFusedActivation activation, dataT* output_data,
    TfLiteType weights_packed_type = kTfLiteNoType) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  bool null_bias = bias_data == nullptr ? true : false;

  constexpr int array_size = 4;  // Avoid variable length array warning.
  const int inputs_size = null_bias ? 2 : 3;
  constexpr int outputs_size = 1;
  const int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[array_size];

  tensors[0] = CreateQuantizedTensor(input_data, input_quantized, input_dims,
                                     input_scale, input_zero_point);
  tensors[1] = CreateQuantizedTensor(
      weights_data, weights_quantized, weights_dims, weights_scale,
      weights_zero_point, false, weights_packed_type);
  if (null_bias) {
    tensors[2] = CreateQuantizedTensor(output_data, output_dims, output_scale,
                                       output_zero_point);
  } else {
    tensors[2] = CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                           input_scale, weights_scale),
    tensors[3] = CreateQuantizedTensor(output_data, output_dims, output_scale,
                                       output_zero_point);
  }

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);

  return ValidateFullyConnectedGoldens(tensors, tensors_size, null_bias,
                                       activation, 0.0f, output_dims_count,
                                       golden_quantized, output_data);
}

#ifdef USE_TFLM_COMPRESSION

template <typename TIO, typename TW, typename TB>
TfLiteStatus TestFullyConnectedQuantizedCompressed(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* output_dims_data,
    const float* expected_output_data, TIO* expected_output_quantized,
    TIO* output_quantized, float output_scale, int output_zero_point,
    const TfLiteFusedActivation activation,
    const TestCompressionQuantizedInfo<TW>* weight_comp_info,
    const TestCompressionQuantizedInfo<TB>* bias_comp_info) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weight_dims = IntArrayFromInts(weight_comp_info->dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_comp_info->dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteFloatArray* weight_scales =
      FloatArrayFromFloats(weight_comp_info->scales);
  TfLiteIntArray* weight_zero_points =
      IntArrayFromInts(weight_comp_info->zero_points);

  TfLiteTensor weight_tensor = CreateQuantizedTensor(
      weight_comp_info->compressed, weight_dims, weight_scales->data[0],
      weight_zero_points->data[0], false, kTfLiteInt8);
  SymmetricQuantize(weight_comp_info->data, weight_comp_info->value_table,
                    weight_comp_info->value_table_stride,
                    weight_scales->data[0]);

  TfLiteTensor bias_tensor = {};
  if (bias_comp_info != nullptr) {
    bias_tensor = CreateQuantizedTensor(bias_comp_info->compressed, bias_dims,
                                        input_scale * weight_scales->data[0], 0,
                                        false, typeToTfLiteType<TB>());
    SymmetricQuantize(bias_comp_info->data, bias_comp_info->value_table,
                      bias_comp_info->value_table_stride,
                      bias_tensor.params.scale);
  }

  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point);

  const int tensors_size =
      (bias_comp_info == nullptr) ? kMaxTensors - 1 : kMaxTensors;
  TfLiteTensor tensors[kMaxTensors] = {};
  tensors[0] = CreateQuantizedTensor(input_data, input_quantized, input_dims,
                                     input_scale, input_zero_point);
  tensors[1] = weight_tensor;
  if (bias_comp_info == nullptr) {
    tensors[2] = output_tensor;
  } else {
    tensors[2] = bias_tensor;
    tensors[3] = output_tensor;
  }

  const int output_dims_count = ElementCount(*output_dims);
  Quantize(expected_output_data, expected_output_quantized, output_dims_count,
           output_scale, output_zero_point);
  return ValidateFullyConnectedGoldens(
      tensors, tensors_size, bias_comp_info == nullptr, activation, 0.0f,
      output_dims_count, expected_output_quantized, output_quantized,
      weight_comp_info, bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

template <typename dataT, typename weightT, typename biasT>
TfLiteStatus TestFullyConnectedQuantizedPerChannel(
    int* input_dims_data, const float* input_data, dataT* input_quantized,
    const float input_scale, const int input_zero_point, int* weights_dims_data,
    const float* weights_data, weightT* weights_quantized,
    float* weights_scales, int* weights_zero_points, int* bias_dims_data,
    const float* bias_data, biasT* bias_quantized, const float* golden,
    dataT* golden_quantized, int* output_dims_data, const float output_scale,
    const int output_zero_point, TfLiteFusedActivation activation,
    dataT* output_data, TfLiteType weights_packed_type = kTfLiteNoType) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  bool null_bias = bias_data == nullptr ? true : false;

  constexpr int array_size = 4;  // Avoid variable length array warning.
  const int inputs_size = null_bias ? 2 : 3;
  constexpr int outputs_size = 1;
  const int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[array_size];
  TfLiteAffineQuantization weights_quant, bias_quant;
  float bias_scales[5];
  int bias_zero_points[5];

  tensors[0] = CreateQuantizedTensor(input_data, input_quantized, input_dims,
                                     input_scale, input_zero_point);
  tensors[1] = CreateSymmetricPerChannelQuantizedTensorWithoutScaleEstimation(
      weights_data, weights_quantized, weights_dims, weights_scales,
      weights_zero_points, &weights_quant, 0 /* quantized dimension */, false,
      weights_packed_type);

  if (null_bias) {
    tensors[2] = CreateQuantizedTensor(output_data, output_dims, output_scale,
                                       output_zero_point);
  } else {
    tensors[2] = CreatePerChannelQuantizedBiasTensor(
        bias_data, bias_quantized, bias_dims, input_scale, &weights_scales[1],
        bias_scales, bias_zero_points, &bias_quant,
        0 /* quantized dimension */);
    tensors[3] = CreateQuantizedTensor(output_data, output_dims, output_scale,
                                       output_zero_point);
  }

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);
  return ValidateFullyConnectedGoldens(
      tensors, tensors_size, null_bias, activation, 1.0f /* tolerance */,
      output_dims_count, golden_quantized, output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTest) {
  float output_data[tflite::testing::simple_output_size];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestCompressed) {
  float output_data[tflite::testing::simple_output_size];

  tflite::testing::TestCompressionInfo<const float> weight_comp_info = {};
  tflite::testing::TestCompressionInfo<const float> bias_comp_info = {};

  weight_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  weight_comp_info.value_table = tflite::testing::kBinQuantWeightValueTable;
  weight_comp_info.value_table_stride =
      tflite::testing::kBinQuantWeightValueTableElements;
  weight_comp_info.bit_width = tflite::testing::kBinQuantWeightBitWidth;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = tflite::testing::simple_bias_data;
  bias_comp_info.value_table_stride = tflite::testing::simple_bias_size;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidth;

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          reinterpret_cast<const float*>(tflite::testing::kBinQuantWeightData),
          tflite::testing::simple_bias_dims,
          reinterpret_cast<const float*>(tflite::testing::kBinQuantBiasData),
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data, &weight_comp_info, &bias_comp_info),
      kTfLiteOk);
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestNullBias) {
  float output_data[tflite::testing::simple_output_size];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, nullptr, nullptr,
          tflite::testing::simple_golden_null_bias,
          tflite::testing::simple_output_dims, kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8Compressed) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  constexpr float weights_scale[] = {1, 1.0f};
  constexpr int weights_zero_point[] = {1, 0};
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::kBinQuantWeightValueTableElements];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> weight_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int32_t> bias_comp_info = {};

  weight_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  weight_comp_info.value_table = weights_quantized;
  weight_comp_info.value_table_stride =
      tflite::testing::kBinQuantWeightValueTableElements;
  weight_comp_info.bit_width = tflite::testing::kBinQuantWeightBitWidth;
  weight_comp_info.compressed = tflite::testing::kBinQuantWeightData;
  weight_comp_info.data = tflite::testing::kBinQuantWeightValueTable;
  weight_comp_info.dims_data = tflite::testing::simple_weights_dims;
  weight_comp_info.scales = weights_scale;
  weight_comp_info.zero_points = weights_zero_point;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride = tflite::testing::simple_bias_size;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidth;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasData;
  bias_comp_info.data = tflite::testing::simple_bias_data;
  bias_comp_info.dims_data = tflite::testing::simple_bias_dims;
  // bias scales and bias zero_points are not used

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantizedCompressed(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_output_dims,
          tflite::testing::simple_golden, golden_quantized, output_data,
          output_scale, output_zero_point, kTfLiteActNone, &weight_comp_info,
          &bias_comp_info),
      kTfLiteOk);
}

#endif  // USE_TFLM_COMPRESSION

#if !defined(HEXAGON)

#if !defined(XTENSA)

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelInt8) {
  const float input_scale = 0.5f;
  const int input_zero_point = -1;
  const float output_scale = 1.0f;
  const int output_zero_point = -1;
  int weights_zero_points[tflite::testing::simple_bias_size + 1] = {
      tflite::testing::simple_bias_size, 0, 0, 0};
  float weights_scales[tflite::testing::simple_bias_size + 1] = {
      tflite::testing::simple_bias_size, 0.2f, 0.25f, 0.5f};

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantizedPerChannel(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scales, weights_zero_points,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          bias_quantized, tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}
#endif  // #if !defined(XTENSA)

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt16) {
  const float input_scale = 128.0 / 65536;
  const int input_zero_point = 0;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;
  const float output_scale = 128.0 / 65536;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int64_t bias_quantized[tflite::testing::simple_output_size];
  int16_t golden_quantized[tflite::testing::simple_output_size];
  int16_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt16Compressed) {
  const float input_scale = 128.0 / 65536;
  const int input_zero_point = 0;
  constexpr float weights_scale[] = {1, 1.0f};
  constexpr int weights_zero_point[] = {1, 0};
  const float output_scale = 128.0 / 65536;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::kBinQuantWeightValueTableElements];
  int64_t bias_quantized[tflite::testing::simple_output_size];
  int16_t golden_quantized[tflite::testing::simple_output_size];
  int16_t output_data[tflite::testing::simple_output_size];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> weight_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int64_t> bias_comp_info = {};

  weight_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  weight_comp_info.value_table = weights_quantized;
  weight_comp_info.value_table_stride =
      tflite::testing::kBinQuantWeightValueTableElements;
  weight_comp_info.bit_width = tflite::testing::kBinQuantWeightBitWidth;
  weight_comp_info.compressed = tflite::testing::kBinQuantWeightData;
  weight_comp_info.data = tflite::testing::kBinQuantWeightValueTable;
  weight_comp_info.dims_data = tflite::testing::simple_weights_dims;
  weight_comp_info.scales = weights_scale;
  weight_comp_info.zero_points = weights_zero_point;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride = tflite::testing::simple_bias_size;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidth;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasData;
  bias_comp_info.data = tflite::testing::simple_bias_data;
  bias_comp_info.dims_data = tflite::testing::simple_bias_dims;
  // bias scales and bias zero_points are not used

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantizedCompressed(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_output_dims,
          tflite::testing::simple_golden, golden_quantized, output_data,
          output_scale, output_zero_point, kTfLiteActNone, &weight_comp_info,
          &bias_comp_info),
      kTfLiteOk);
}

#endif  // USE_TFLM_COMPRESSION

#if !defined(XTENSA) && !defined(CMSIS_NN)

TF_LITE_MICRO_TEST(SimpleTestPerChannelQuantizedInt16) {
  const float input_scale = 128.0 / 65536;
  const int input_zero_point = 0;
  const float output_scale = 128.0 / 65536;
  const int output_zero_point = 0;
  int weights_zero_points[tflite::testing::simple_bias_size + 1] = {
      tflite::testing::simple_bias_size, 0, 0, 0};
  float weights_scales[tflite::testing::simple_bias_size + 1] = {
      tflite::testing::simple_bias_size, 0.2f, 0.25f, 0.5f};

  int16_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int64_t bias_quantized[tflite::testing::simple_output_size];
  int16_t golden_quantized[tflite::testing::simple_output_size];
  int16_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantizedPerChannel(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scales, weights_zero_points,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          bias_quantized, tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

#endif  // !defined(XTENSA) && !defined(CMSIS_NN)

#endif  // !defined(HEXAGON)

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;

  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int input_dims_4d[] = {4, 1, 1, 2, 10};

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          input_dims_4d, tflite::testing::simple_input_data, input_quantized,
          input_scale, input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8Relu) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;

  const float output_scale = 0.5f;
  const int output_zero_point = -128;

  int8_t input_quantized[tflite::testing::relu_input_size];
  int8_t weights_quantized[tflite::testing::relu_weights_size];
  int32_t bias_quantized[tflite::testing::relu_output_size];
  int8_t golden_quantized[tflite::testing::relu_output_size];
  int8_t output_data[tflite::testing::relu_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::relu_input_dims, tflite::testing::relu_input_data,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::relu_weights_dims,
          tflite::testing::relu_weights_data, weights_quantized, weights_scale,
          weights_zero_point, tflite::testing::relu_bias_dims,
          tflite::testing::relu_bias_data, bias_quantized,
          tflite::testing::relu_golden, golden_quantized,
          tflite::testing::relu_output_dims, output_scale, output_zero_point,
          kTfLiteActRelu, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInput) {
  int input_dims_4d[] = {4, 1, 1, 2, 10};

  float output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_4d, tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(Representative1x64Input1x16Output) {
  float output_data[tflite::testing::representative_64x16_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::representative_64x16_input_dims,
          tflite::testing::representative_64x16_input_data,
          tflite::testing::representative_64x16_weights_dims,
          tflite::testing::representative_64x16_weights_data,
          tflite::testing::representative_64x16_bias_dims,
          tflite::testing::representative_64x16_bias_data,
          tflite::testing::representative_64x16_golden,
          tflite::testing::representative_64x16_output_dims, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(Representative1x64Input1x16OutputQuantizedInt8) {
  const float input_scale = 0.051445;
  const int input_zero_point = -128;
  const float weights_scale = 0.005660;
  const int weights_zero_point = 0;

  const float output_scale = 0.069785;
  const int output_zero_point = -9;

  int8_t input_quantized[tflite::testing::representative_64x16_input_size];
  int8_t weights_quantized[tflite::testing::representative_64x16_weights_size];
  int32_t bias_quantized[tflite::testing::representative_64x16_output_size];
  int8_t golden_quantized[tflite::testing::representative_64x16_output_size];
  int8_t output_data[tflite::testing::representative_64x16_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::representative_64x16_input_dims,
          tflite::testing::representative_64x16_input_data, input_quantized,
          input_scale, input_zero_point,
          tflite::testing::representative_64x16_weights_dims,
          tflite::testing::representative_64x16_weights_data, weights_quantized,
          weights_scale, weights_zero_point,
          tflite::testing::representative_64x16_bias_dims,
          tflite::testing::representative_64x16_bias_data, bias_quantized,
          tflite::testing::representative_64x16_golden, golden_quantized,
          tflite::testing::representative_64x16_output_dims, output_scale,
          output_zero_point, kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8NullBias) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, nullptr, nullptr,
          static_cast<int32_t*>(nullptr),
          tflite::testing::simple_golden_null_bias, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

// TODO(b/258710417): INT4 isn't currently supported on Hexagon.
#if !defined(HEXAGON)
// This test was created by handcrafting simple_int4_weights_data, and
// simple_golden_null_bias_int4_weights was obtained by running
// TestFullyConnectedQuantized() with int8 quantization, and ensuring that int4
// quantization yields the same outputs.
TF_LITE_MICRO_TEST(SimpleTestQuantizedInt4Weights) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_int4_weights_data, weights_quantized,
          weights_scale, weights_zero_point, nullptr, nullptr,
          static_cast<int32_t*>(nullptr),
          tflite::testing::simple_golden_null_bias_int4_weights,
          golden_quantized, tflite::testing::simple_output_dims, output_scale,
          output_zero_point, kTfLiteActNone, output_data, kTfLiteInt4),
      kTfLiteOk);
}
#endif  // !defined(HEXAGON)

TF_LITE_MICRO_TESTS_END
