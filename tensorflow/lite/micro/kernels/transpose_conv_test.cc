/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/transpose_conv.h"

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Common inputs and outputs.
constexpr int kInputElements = 32;
static int kInputShape[] = {4, 1, 4, 4, 2};
static const float kInputData[kInputElements] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

constexpr int kFilterElements = 18;
static int kFilterShape[] = {4, 1, 3, 3, 2};
static const float kFilterData[kFilterElements] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

constexpr int kBiasElements = 1;
static int kBiasShape[] = {4, 1, 1, 1, 1};
static const float kBiasData[kBiasElements] = {0};

constexpr int kOutputElements = 16;
static int kOutputShape[] = {4, 1, 4, 4, 1};
static const float kGoldenData[kOutputElements] = {
    184,  412,  568,  528,  678,  1347, 1689, 1434,
    1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760};

#ifdef USE_TFLM_COMPRESSION

constexpr size_t kTransposeConvMaxTensors = 5;
constexpr size_t kTransposeConvMaxInputTensors = 4;

// compressed filter data for kBinQuant scheme, matches kFilterData
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantFilterData[] = {
    0x00, 0x44, 0x32, 0x14, 0xC7, 0x42, 0x54, 0xB6, 0x35, 0xCF, 0x84, 0x40};
constexpr int kBinQuantFilterBitWidth = 5;
// compressed bias data for kBinQuant scheme, matches kBiasData
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantBiasData[] = {0x00};
constexpr int kBinQuantBiasBitWidth = 1;

// Common inputs and outputs (quantized single channel).
// data from TfLite test: SimpleBiasTestQuantizedPerChannelSingleChannel
static int kInputShapeQ1[] = {4, 1, 4, 4, 1};
static constexpr float kInputDataQ1[] = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
constexpr size_t kInputElementsQ1 = std::extent<decltype(kInputDataQ1)>::value;

constexpr int kNumChannelsQ1 = 1;
static int kFilterShapeQ1[] = {4, 1, 3, 3, 1};
static constexpr float kFilterDataQ1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
constexpr size_t kFilterElementsQ1 =
    std::extent<decltype(kFilterDataQ1)>::value;

static int kBiasShapeQ1[] = {1, 1};
static constexpr float kBiasDataQ1[] = {1};
constexpr size_t kBiasElementsQ1 = std::extent<decltype(kBiasDataQ1)>::value;

static int kOutputShapeQ1[] = {4, 1, 4, 4, 1};
static constexpr float kGoldenDataQ1[] = {
    30, 62, 84, 76, 100, 194, 238, 200, 208, 372, 418, 330, 264, 446, 486, 366};
constexpr int kOutputElementsQ1 = std::extent<decltype(kGoldenDataQ1)>::value;

// compressed filter data for kBinQuant scheme, matches kFilterDataQ1
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantFilterDataQ1[] = {0x01, 0x23, 0x45, 0x67,
                                                         0x80};
constexpr int kBinQuantFilterBitWidthQ1 = 4;
// compressed bias data for kBinQuant scheme, matches kBiasDataQ1
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantBiasDataQ1[] = {0x00};
constexpr int kBinQuantBiasBitWidthQ1 = 1;

// Common inputs and outputs (quantized multi channel).
// data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
static int kInputShapeQ2[] = {4, 1, 2, 3, 2};
static constexpr float kInputDataQ2[] = {
    // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
    3,  2,   // batch = 0, y = 0, x = 0
    1,  -1,  // batch = 0, y = 0, x = 1
    -2, -3,  // batch = 0, y = 0, x = 2
    4,  3,   // batch = 0, y = 1, x = 0
    2,  -2,  // batch = 0, y = 1, x = 1
    -3, -4,  // batch = 0, y = 1, x = 2
};
constexpr size_t kInputElementsQ2 = std::extent<decltype(kInputDataQ2)>::value;

constexpr int kNumChannelsQ2 = 2;
static int kFilterShapeQ2[] = {4, 2, 2, 2, 2};
// Original filter data:
// static constexpr float kFilterDataQ2[] = {
//     // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
//     1, 2,  // out channel = 0, y = 0, x = 0
//     3, 4,  // out channel = 0, y = 0, x = 1
//     3, 4,  // out channel = 0, y = 1, x = 0
//     5, 6,  // out channel = 0, y = 1, x = 1
//     7, 8,  // out channel = 1, y = 0, x = 0
//     5, 6,  // out channel = 1, y = 0, x = 1
//     3, 4,  // out channel = 1, y = 1, x = 0
//     1, 2,  // out channel = 1, y = 1, x = 1
// };

static int kBiasShapeQ2[] = {1, 2};
static constexpr float kBiasDataQ2[] = {3, -2};
constexpr size_t kBiasElementsQ2 = std::extent<decltype(kBiasDataQ2)>::value;

static int kOutputShapeQ2[] = {4, 1, 2, 3, 2};
static constexpr float kGoldenDataQ2[] = {10, 35, 19, 24, -6,  -41,
                                          30, 64, 51, 40, -29, -64};
constexpr int kOutputElementsQ2 = std::extent<decltype(kGoldenDataQ2)>::value;

// compressed filter data for kBinQuant scheme, matches kFilterDataQ2
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantFilterDataQ2[] = {0x05, 0x34, 0xE5,
                                                         0xDE, 0x54, 0xC1};
constexpr float kBinQuantFilterValueTableQ2[] = {1, 2, 3, 4, 5, 6, 0, 0,
                                                 1, 2, 3, 4, 5, 6, 7, 8};
constexpr size_t kBinQuantFilterValueTableElementsQ2 =
    std::extent<decltype(kBinQuantFilterValueTableQ2)>::value;
constexpr int kBinQuantFilterBitWidthQ2 = 3;
// compressed bias data for kBinQuant scheme, matches kBiasDataQ2
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantBiasDataQ2[] = {0x00};
constexpr int kBinQuantBiasBitWidthQ2 = 1;

#endif  // USE_TFLM_COMPRESSION

// Transpose conv uses TfLiteConvParams.
static const TfLiteConvParams common_conv_params = {
    kTfLitePaddingSame,  // padding
    1,                   // stride_width
    1,                   // stride_height
    kTfLiteActNone,
    1,
    1,
    kTfLiteNoType};

template <typename T>
TfLiteStatus InvokeTransposeConv(
    TfLiteTensor* tensors, int tensors_size, int output_length,
    const TfLiteConvParams* conv_params, T* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const CompressedTensorList* comp_list_p = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  // TODO(b/358151309): support optional bias tensor
  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_TRANSPOSE_CONV();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, conv_params
#ifdef USE_TFLM_COMPRESSION
                             ,
                             nullptr, comp_list_p
#endif  // USE_TFLM_COMPRESSION
  );

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  return runner.Invoke();
}

template <typename T, typename TF = void, typename TB = void>
TfLiteStatus ValidateTransposeConvGoldens(
    TfLiteTensor* tensors, int tensors_size, const T* expected_output_data,
    int output_length, const TfLiteConvParams* conv_params, T* output_data,
    float tolerance = 1e-5f
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<TF>* filter_comp_info = nullptr,
    const TestCompressionInfo<TB>* bias_comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
#ifdef USE_TFLM_COMPRESSION

  TestCompressedList<kTransposeConvMaxInputTensors> tcl;
  if (filter_comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(
        tcl.AddInput(*filter_comp_info, tensors[kTransposeConvFilterTensor],
                     kTransposeConvFilterTensor),
        kTfLiteOk);
    TF_LITE_MICRO_CHECK_FAIL();
  }
  if (bias_comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(
        tcl.AddInput(*bias_comp_info, tensors[kTransposeConvBiasTensor],
                     kTransposeConvBiasTensor),
        kTfLiteOk);
    TF_LITE_MICRO_CHECK_FAIL();
  }
  const CompressedTensorList* comp_list_p = tcl.GetCompressedTensorList();

#endif  // USE_TFLM_COMPRESSION

  TfLiteStatus status = InvokeTransposeConv(
      tensors, tensors_size, output_length, conv_params, output_data
#ifdef USE_TFLM_COMPRESSION
      ,
      comp_list_p
#endif  // USE_TFLM_COMPRESSION
  );
  if (status != kTfLiteOk) {
    return status;
  }
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

TfLiteStatus TestTransposeConvFloat(
    int* input_dims_data, const float* input_data, int* filter_dims_data,
    const float* filter_data, int* bias_dims_data, const float* bias_data,
    int* output_dims_data, const float* expected_output_data,
    const TfLiteConvParams* conv_params, float* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<const float>* filter_comp_info = nullptr,
    const TestCompressionInfo<const float>* bias_comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(filter_data, filter_dims),
      CreateTensor(input_data, input_dims),
      CreateTensor(bias_data, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  return ValidateTransposeConvGoldens(tensors, tensors_size,
                                      expected_output_data, output_dims_count,
                                      conv_params, output_data
#ifdef USE_TFLM_COMPRESSION
                                      ,
                                      1e-5, filter_comp_info, bias_comp_info
#endif  // USE_TFLM_COMPRESSION
  );
}

TfLiteStatus TestTransposeConvQuantized(
    int* input_dims_data, const float* input_data, int8_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_quantized, float filter_scale,
    int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_quantized,
    float output_scale, int output_zero_point,
    const TfLiteConvParams* conv_params, int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  tflite::Quantize(expected_output_data, expected_output_quantized,
                   output_dims_count, output_scale, 0);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims), filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, filter_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_quantized, output_dims_count,
      conv_params, output_data, 1.0f);
}

template <typename T>
TfLiteStatus TestTransposeConvQuantized(
    int* input_dims_data, const float* input_data, int16_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_quantized, float filter_scale,
    int* bias_dims_data, const float* bias_data, T* bias_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int16_t* expected_output_quantized,
    float output_scale, int output_zero_point,
    const TfLiteConvParams* conv_params, int16_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  tflite::Quantize(expected_output_data, expected_output_quantized,
                   output_dims_count, output_scale, 0);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims), filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, filter_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  // Tolerance is slightly looser for 8x16 compared with float, since quant
  // error is more pronounced on the finer-grained 16-bit output.
  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_quantized, output_dims_count,
      conv_params, output_data, 4.0f);
}

#ifdef USE_TFLM_COMPRESSION

template <typename TIO, typename TBIAS>
TfLiteStatus TestTransposeConvQuantizedCompressed(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* output_dims_data,
    const float* expected_output_data, TIO* expected_output_quantized,
    TIO* output_quantized, float output_scale, int output_zero_point,
    const TfLiteConvParams* conv_params, const unsigned int tolerance,
    const TestCompressionQuantizedInfo<int8_t>* filter_comp_info,
    const TestCompressionQuantizedInfo<TBIAS>* bias_comp_info) {
  // TODO(b/358151309): account for optional bias tensor
  // bool null_bias = comp_info->bias_data == nullptr ? true : false;

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_comp_info->dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_comp_info->dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteFloatArray* filter_scales =
      FloatArrayFromFloats(filter_comp_info->scales);
  TfLiteIntArray* filter_zero_points =
      IntArrayFromInts(filter_comp_info->zero_points);
  TfLiteFloatArray* bias_scales = FloatArrayFromFloats(bias_comp_info->scales);
  TfLiteIntArray* bias_zero_points =
      IntArrayFromInts(bias_comp_info->zero_points);

  TfLiteAffineQuantization filter_quant = {};
  TfLiteTensor filter_tensor = CreatePerChannelQuantizedTensor(
      filter_comp_info->compressed, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, kTransposeConvQuantizedDimension,
      false /* is_variable */, kTfLiteInt8);
  SymmetricPerChannelQuantize(
      filter_comp_info->data, filter_comp_info->value_table,
      filter_scales->size * filter_comp_info->value_table_stride,
      filter_scales->size, filter_scales->data);

  TfLiteAffineQuantization bias_quant = {};
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_comp_info->compressed, bias_dims, input_scale, filter_scales,
      bias_scales, bias_zero_points, &bias_quant,
      kTransposeConvQuantizedDimension, false /* is_variable */,
      typeToTfLiteType<TBIAS>());
  SymmetricPerChannelQuantize(
      bias_comp_info->data, bias_comp_info->value_table,
      bias_scales->size * bias_comp_info->value_table_stride, bias_scales->size,
      bias_scales->data);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kTransposeConvMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      bias_tensor,
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  const int output_dims_count = ElementCount(*output_dims);
  Quantize(expected_output_data, expected_output_quantized, output_dims_count,
           output_scale, output_zero_point);
  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_quantized, output_dims_count,
      conv_params, output_quantized, tolerance, filter_comp_info,
      bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTestFloat) {
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          tflite::testing::kBiasShape, tflite::testing::kBiasData,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          &tflite::testing::common_conv_params, output_data));
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestFloatCompressed) {
  tflite::testing::TestCompressionInfo<const float> filter_comp_info = {};
  tflite::testing::TestCompressionInfo<const float> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = tflite::testing::kFilterData;
  filter_comp_info.value_table_stride =
      std::extent<decltype(tflite::testing::kFilterData)>::value;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidth;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = tflite::testing::kBiasData;
  bias_comp_info.value_table_stride =
      std::extent<decltype(tflite::testing::kBiasData)>::value;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidth;

  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          tflite::testing::kFilterShape,
          reinterpret_cast<const float*>(tflite::testing::kBinQuantFilterData),
          tflite::testing::kBiasShape,
          reinterpret_cast<const float*>(tflite::testing::kBinQuantBiasData),
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          &tflite::testing::common_conv_params, output_data, &filter_comp_info,
          &bias_comp_info));
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(fusedRELUTest) {
  float output_data[tflite::testing::kOutputElements];
  float golden_data[] = {29,  24,  0, 0, 99,  72,  0,   0,
                         207, 186, 0, 0, 263, 292, 141, 0};
  int filter_shape[] = {4, 1, 3, 3, 1};
  float filter_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int input_shape[] = {4, 1, 4, 4, 1};
  float input_data[] = {1, 2,  -3,  -4,  5,  6,  -7, -8,
                        9, 10, -11, -12, 13, 14, 15, 16};
  TfLiteConvParams conv_params = {kTfLitePaddingSame,  // padding
                                  1,                   // stride_width
                                  1,                   // stride_height
                                  kTfLiteActRelu,
                                  1,
                                  1,
                                  kTfLiteNoType};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestTransposeConvFloat(
                     input_shape, input_data, filter_shape, filter_data,
                     tflite::testing::kBiasShape, tflite::testing::kBiasData,
                     tflite::testing::kOutputShape, golden_data, &conv_params,
                     output_data));
}

TF_LITE_MICRO_TEST(AccuracyWithFusedActivationTest) {
  int output_shape[] = {4, 1, 3, 4, 1};
  float output_data[tflite::testing::kOutputElements];
  float golden_data[] = {1615, 1938, 0, 0, 2584, 1615, 0, 0, 323, 1292, 0, 0};
  int filter_shape[] = {4, 1, 3, 3, 1};
  float filter_data[] = {9, 5, 6, 9, 8, 5, 3, 1, 4};
  int input_shape[] = {4, 1, 1, 2, 1};
  float input_data[] = {323, -521};
  TfLiteConvParams conv_params = {kTfLitePaddingSame,  // padding
                                  3,                   // stride_width
                                  3,                   // stride_height
                                  kTfLiteActRelu,
                                  1,
                                  1,
                                  kTfLiteNoType};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestTransposeConvFloat(
                     input_shape, input_data, filter_shape, filter_data,
                     tflite::testing::kBiasShape, tflite::testing::kBiasData,
                     output_shape, golden_data, &conv_params, output_data));
}

TF_LITE_MICRO_TEST(MultiChannelBiasWithFusedActivationTest) {
  int output_shape[] = {4, 1, 5, 5, 2};
  float output_data[50];
  float golden_data[] = {4,  6,  6,  8,  10, 14, 9,  12, 13, 16, 10, 12, 12,
                         14, 28, 32, 21, 24, 25, 28, 13, 12, 9,  8,  35, 40,
                         45, 52, 57, 64, 0,  0,  0,  0,  0,  0,  39, 44, 47,
                         52, 0,  0,  0,  0,  4,  6,  63, 68, 71, 76};
  int filter_shape[] = {4, 2, 3, 3, 1};
  float filter_data[] = {1, 3, 5, 7, 9,  11, 13, 15, 17,
                         2, 4, 6, 8, 10, 12, 14, 16, 18};
  int input_shape[] = {4, 1, 2, 2, 1};
  float input_data[] = {1, 2, -3, 4};
  int bias_shape[] = {4, 2, 1, 1, 1};
  float bias_data[] = {3, 4};
  TfLiteConvParams conv_params = {kTfLitePaddingValid,  // padding
                                  2,                    // stride_width
                                  2,                    // stride_height
                                  kTfLiteActRelu,
                                  1,
                                  1,
                                  kTfLiteNoType};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          input_shape, input_data, filter_shape, filter_data, bias_shape,
          bias_data, output_shape, golden_data, &conv_params, output_data));
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannel) {
  int8_t output_data[tflite::testing::kOutputElements];

  const float input_scale = 0.5f;
  const float output_scale = 30.0f;
  const float filter_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, filter_scale, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized, scales, zero_points,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          golden_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, output_data));
}

TF_LITE_MICRO_TEST(SimpleTestQuantized16x8PerChannel) {
  int16_t output_data[tflite::testing::kOutputElements];

  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const float filter_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  std::int64_t bias_quantized[tflite::testing::kBiasElements];
  int16_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, filter_scale, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized, scales, zero_points,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          golden_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, output_data));
}

TF_LITE_MICRO_TEST(SimpleTestQuantized16x8PerChannelWithInt16Bias) {
  int16_t output_data[tflite::testing::kOutputElements];

  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const float filter_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int16_t bias_quantized[tflite::testing::kBiasElements];
  int16_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, filter_scale, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized, scales, zero_points,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          golden_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, output_data));
}

TF_LITE_MICRO_TEST(InputOutputDifferentTypeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  const int output_dims_count = tflite::ElementCount(*output_dims);
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t output_data[tflite::testing::kOutputElements];

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateTensor(tflite::testing::kFilterData, filter_dims),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
  };
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::InvokeTransposeConv(
                        tensors, tensors_size, output_dims_count,
                        &tflite::testing::common_conv_params, output_data));
}

TF_LITE_MICRO_TEST(HybridModeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  const int output_dims_count = tflite::ElementCount(*output_dims);

  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t filter_data[tflite::testing::kFilterElements] = {};
  float output_data[tflite::testing::kOutputElements];

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateQuantizedTensor(filter_data, filter_dims,
                            /*scale=*/1.0f,
                            /*zero_point=*/0),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::InvokeTransposeConv(
                        tensors, tensors_size, output_dims_count,
                        &tflite::testing::common_conv_params, output_data));
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelSingleChannelCompressed) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannelSingleChannel
  const float input_scale = 16.0f / 255.0f;
  const float output_scale = 2.0f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;
  constexpr float filter_scales[] = {
      tflite::testing::kNumChannelsQ1,
      9.0f / 127.0f,
  };
  constexpr int filter_zero_points[] = {
      tflite::testing::kNumChannelsQ1,
      0,
  };
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int8_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ1];
  int32_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int8_t golden_quantized[tflite::testing::kOutputElementsQ1];
  int8_t output_quantized[tflite::testing::kOutputElementsQ1];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int32_t> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = filter_quantized;
  filter_comp_info.value_table_stride =
      tflite::testing::kFilterElementsQ1 / tflite::testing::kNumChannelsQ1;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidthQ1;
  filter_comp_info.compressed = tflite::testing::kBinQuantFilterDataQ1;
  filter_comp_info.data = tflite::testing::kFilterDataQ1;
  filter_comp_info.dims_data = tflite::testing::kFilterShapeQ1;
  filter_comp_info.scales = filter_scales;
  filter_comp_info.zero_points = filter_zero_points;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride =
      tflite::testing::kBiasElementsQ1 / tflite::testing::kNumChannelsQ1;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidthQ1;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasDataQ1;
  bias_comp_info.data = tflite::testing::kBiasDataQ1;
  bias_comp_info.dims_data = tflite::testing::kBiasShapeQ1;
  bias_comp_info.scales = bias_scales;
  bias_comp_info.zero_points = bias_zero_points;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1,
          golden_quantized, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, 0, &filter_comp_info,
          &bias_comp_info));
}

TF_LITE_MICRO_TEST(
    SimpleBiasTestQuantizedPerChannelBias16MultiChannelCompressed) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
  const float input_scale = 4.0f / 127.0f;
  const float output_scale = 128.0f / 65536.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;
  constexpr float filter_scales[] = {
      tflite::testing::kNumChannelsQ2,
      7.0f / 127.0f,
      8.0f / 127.0f,
  };
  constexpr int filter_zero_points[] = {
      tflite::testing::kNumChannelsQ2,
      0,
      0,
  };
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ2];
  int8_t filter_quantized[tflite::testing::kBinQuantFilterValueTableElementsQ2];
  int16_t bias_quantized[tflite::testing::kBiasElementsQ2];
  int16_t golden_quantized[tflite::testing::kOutputElementsQ2];
  int16_t output_quantized[tflite::testing::kOutputElementsQ2];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int16_t> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = filter_quantized;
  filter_comp_info.value_table_stride =
      tflite::testing::kBinQuantFilterValueTableElementsQ2 /
      tflite::testing::kNumChannelsQ2;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidthQ2;
  filter_comp_info.compressed = tflite::testing::kBinQuantFilterDataQ2;
  filter_comp_info.data = tflite::testing::kBinQuantFilterValueTableQ2;
  filter_comp_info.dims_data = tflite::testing::kFilterShapeQ2;
  filter_comp_info.scales = filter_scales;
  filter_comp_info.zero_points = filter_zero_points;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride =
      tflite::testing::kBiasElementsQ2 / tflite::testing::kNumChannelsQ2;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidthQ2;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasDataQ2;
  bias_comp_info.data = tflite::testing::kBiasDataQ2;
  bias_comp_info.dims_data = tflite::testing::kBiasShapeQ2;
  bias_comp_info.scales = bias_scales;
  bias_comp_info.zero_points = bias_zero_points;

  // The quantized output is compared to the expected output (quantized).
  // A tolerance of 81 is approx. 0.1582f which is less than the TfLite
  // tolerance of 0.19f.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShapeQ2, tflite::testing::kInputDataQ2,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ2, tflite::testing::kGoldenDataQ2,
          golden_quantized, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, 81, &filter_comp_info,
          &bias_comp_info));
}

TF_LITE_MICRO_TEST(
    SimpleBiasTestQuantizedPerChannelBias64MultiChannelCompressed) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
  const float input_scale = 4.0f / 127.0f;
  const float output_scale = 128.0f / 65536.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;
  constexpr float filter_scales[] = {
      tflite::testing::kNumChannelsQ2,
      7.0f / 127.0f,
      8.0f / 127.0f,
  };
  constexpr int filter_zero_points[] = {
      tflite::testing::kNumChannelsQ2,
      0,
      0,
  };
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ2];
  int8_t filter_quantized[tflite::testing::kBinQuantFilterValueTableElementsQ2];
  int64_t bias_quantized[tflite::testing::kBiasElementsQ2];
  int16_t golden_quantized[tflite::testing::kOutputElementsQ2];
  int16_t output_quantized[tflite::testing::kOutputElementsQ2];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int64_t> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = filter_quantized;
  filter_comp_info.value_table_stride =
      tflite::testing::kBinQuantFilterValueTableElementsQ2 /
      tflite::testing::kNumChannelsQ2;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidthQ2;
  filter_comp_info.compressed = tflite::testing::kBinQuantFilterDataQ2;
  filter_comp_info.data = tflite::testing::kBinQuantFilterValueTableQ2;
  filter_comp_info.dims_data = tflite::testing::kFilterShapeQ2;
  filter_comp_info.scales = filter_scales;
  filter_comp_info.zero_points = filter_zero_points;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride =
      tflite::testing::kBiasElementsQ2 / tflite::testing::kNumChannelsQ2;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidthQ2;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasDataQ2;
  bias_comp_info.data = tflite::testing::kBiasDataQ2;
  bias_comp_info.dims_data = tflite::testing::kBiasShapeQ2;
  bias_comp_info.scales = bias_scales;
  bias_comp_info.zero_points = bias_zero_points;

  // The quantized output is compared to the expected output (quantized).
  // A tolerance of 81 is approx. 0.1582f which is less than the TfLite
  // tolerance of 0.19f.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShapeQ2, tflite::testing::kInputDataQ2,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ2, tflite::testing::kGoldenDataQ2,
          golden_quantized, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, 81, &filter_comp_info,
          &bias_comp_info));
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TESTS_END
