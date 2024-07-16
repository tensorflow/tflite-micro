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

#include <cstdint>
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

constexpr float kTolerance = 1e-5;

// Common inputs and outputs.
constexpr int kInputElements = 32;
static int kInputShape[] = {4, 1, 4, 4, 2};
static constexpr float kInputData[kInputElements] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

constexpr int kFilterElements = 18;
static int kFilterShape[] = {4, 1, 3, 3, 2};
static constexpr float kFilterData[kFilterElements] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

constexpr int kBiasElements = 1;
static int kBiasShape[] = {4, 1, 1, 1, 1};
static constexpr float kBiasData[kBiasElements] = {0};

constexpr int kOutputElements = 16;
static int kOutputShape[] = {4, 1, 4, 4, 1};
static constexpr float kGoldenData[kOutputElements] = {
    184,  412,  568,  528,  678,  1347, 1689, 1434,
    1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760};

// Common inputs and outputs (quantized single channel).
// data from TfLite test: SimpleBiasTestQuantizedPerChannelSingleChannel
constexpr int kInputElementsQ1 = 16;
static int kInputShapeQ1[] = {4, 1, 4, 4, 1};
static constexpr float kInputDataQ1[kInputElementsQ1] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

constexpr int kFilterElementsQ1 = 9;
static int kFilterShapeQ1[] = {4, 1, 3, 3, 1};
static constexpr float kFilterDataQ1[kFilterElementsQ1] = {1, 2, 3, 4, 5,
                                                           6, 7, 8, 9};

constexpr int kBiasElementsQ1 = 1;
static int kBiasShapeQ1[] = {1, 1};
static constexpr float kBiasDataQ1[kBiasElementsQ1] = {1};

constexpr int kOutputElementsQ1 = 16;
static int kOutputShapeQ1[] = {4, 1, 4, 4, 1};
static constexpr float kGoldenDataQ1[kOutputElementsQ1] = {
    30, 62, 84, 76, 100, 192, 238, 198, 206, 372, 416, 330, 262, 446, 484, 366};

// Common inputs and outputs (quantized multi channel).
// data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
constexpr int kInputElementsQ2 = 12;
static int kInputShapeQ2[] = {4, 1, 2, 3, 2};
static constexpr float kInputDataQ2[kInputElementsQ2] = {
    // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
    3,  2,   // batch = 0, y = 0, x = 0
    1,  -1,  // batch = 0, y = 0, x = 1
    -2, -3,  // batch = 0, y = 0, x = 2
    4,  3,   // batch = 0, y = 1, x = 0
    2,  -2,  // batch = 0, y = 1, x = 1
    -3, -4,  // batch = 0, y = 1, x = 2
};

constexpr int kFilterElementsQ2 = 16;
static int kFilterShapeQ2[] = {4, 2, 2, 2, 2};
static constexpr float kFilterDataQ2[kFilterElementsQ2] = {
    // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
    1, 2,  // out channel = 0, y = 0, x = 0
    3, 4,  // out channel = 0, y = 0, x = 1
    3, 4,  // out channel = 0, y = 1, x = 0
    5, 6,  // out channel = 0, y = 1, x = 1
    7, 8,  // out channel = 1, y = 0, x = 0
    5, 6,  // out channel = 1, y = 0, x = 1
    3, 4,  // out channel = 1, y = 1, x = 0
    1, 2,  // out channel = 1, y = 1, x = 1
};

constexpr int kBiasElementsQ2 = 2;
static int kBiasShapeQ2[] = {1, 2};
static constexpr float kBiasDataQ2[kBiasElementsQ2] = {3, -2};

constexpr int kOutputElementsQ2 = 12;
static int kOutputShapeQ2[] = {4, 1, 2, 3, 2};
static constexpr float kGoldenDataQ2[kOutputElementsQ2] = {
    10, 35, 19, 24, -6, -41, 30, 64, 51, 40, -29, -64};

// Transpose conv uses TfLiteConvParams.
static TfLiteConvParams common_conv_params = {kTfLitePaddingSame,  // padding
                                              1,  // stride_width
                                              1,  // stride_height
                                              kTfLiteActNone,
                                              1,
                                              1,
                                              kTfLiteNoType};

// Compression inputs and associated data
constexpr int kMaxTensors = 5;
constexpr int kOutputTensor = 4;  // physical index

#ifdef USE_TFLM_COMPRESSION

constexpr int kFilterTensor = 1;  // physical index
constexpr int kBiasTensor = 3;    // physical index

template <typename TFILTER, typename TBIAS>
struct TestCompressionInfo {
  TFILTER* filter_value_table;
  size_t filter_value_table_stride;
  int filter_bit_width;
  bool use_filter_alt_axis;
  TBIAS* bias_value_table;
  size_t bias_value_table_stride;
  int bias_bit_width;
  bool use_bias_alt_axis;
  CompressionScheme scheme;
};

template <typename TBIAS>
struct TestCompressionQuantizedInfo : TestCompressionInfo<int8_t, TBIAS> {
  const uint8_t* filter_compressed;
  const float* filter_data;
  const int* filter_dims_data;    // TfLiteIntArray
  const float* filter_scales;     // TfLiteFloatArray
  const int* filter_zero_points;  // TfLiteIntArray

  const uint8_t* bias_compressed;
  const float* bias_data;
  const int* bias_dims_data;  // TfLiteIntArray
  float* bias_scales;         // TfLiteFloatArray (computed)
  int* bias_zero_points;      // TfLiteIntArray (computed)
};

#endif  // USE_TFLM_COMPRESSION

template <typename CTF = void, typename CTB = void>
TfLiteStatus InvokeTransposeConv(
    TfLiteTensor* tensors, int tensors_size, const TfLiteConvParams* conv_params
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  // TODO(ddavis-2015): account for optional bias tensor

#ifdef USE_TFLM_COMPRESSION

  CompressionTensorData* compressed_tensors[kMaxTensors] = {};
  CompressionTensorData filter_comp_data = {};
  CompressionTensorData bias_comp_data = {};
  CompressedTensorList comp_list = {compressed_tensors};
  CompressedTensorList* comp_list_p = nullptr;

  if (comp_info != nullptr) {
    if (comp_info->scheme == CompressionScheme::kBinQuant) {
      bool is_per_channel_quantized =
          std::is_same<CTF, float>::value ? false : true;
      if (comp_info->filter_value_table != nullptr) {
        compressed_tensors[kFilterTensor] = &filter_comp_data;
        filter_comp_data.scheme = CompressionScheme::kBinQuant;
        filter_comp_data.data.bin_quant.compressed_bit_width =
            comp_info->filter_bit_width;
        filter_comp_data.data.bin_quant.value_table =
            comp_info->filter_value_table;
        filter_comp_data.data.bin_quant.value_table_channel_stride =
            comp_info->filter_value_table_stride;
        filter_comp_data.data.bin_quant.is_per_channel_quantized =
            is_per_channel_quantized;
        filter_comp_data.data.bin_quant.use_alternate_axis =
            comp_info->use_filter_alt_axis;
      }
      if (comp_info->bias_value_table != nullptr) {
        compressed_tensors[kBiasTensor] = &bias_comp_data;
        bias_comp_data.scheme = CompressionScheme::kBinQuant;
        bias_comp_data.data.bin_quant.compressed_bit_width =
            comp_info->bias_bit_width;
        bias_comp_data.data.bin_quant.value_table = comp_info->bias_value_table;
        bias_comp_data.data.bin_quant.value_table_channel_stride =
            comp_info->bias_value_table_stride;
        bias_comp_data.data.bin_quant.is_per_channel_quantized =
            is_per_channel_quantized;
        bias_comp_data.data.bin_quant.use_alternate_axis =
            comp_info->use_bias_alt_axis;
      }
      comp_list_p = &comp_list;
    } else {
      return kTfLiteError;
    }
  }

#endif  // USE_TFLM_COMPRESSION

  const TFLMRegistration registration = tflite::Register_TRANSPOSE_CONV();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, conv_params, nullptr
#ifdef USE_TFLM_COMPRESSION
                             ,
                             comp_list_p
#endif  // USE_TFLM_COMPRESSION
  );

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  return runner.Invoke();
}

template <typename T, typename CTF = void, typename CTB = void>
TfLiteStatus ValidateTransposeConvGoldens(
    TfLiteTensor* tensors, int tensors_size, const float* expected_output_data,
    int output_length, float* output_data, T* output_quantized,
    TfLiteConvParams* conv_params, float tolerance
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  TfLiteStatus status = InvokeTransposeConv(tensors, tensors_size, conv_params
#ifdef USE_TFLM_COMPRESSION
                                            ,
                                            comp_info
#endif  // USE_TFLM_COMPRESSION
  );
  if (status != kTfLiteOk) {
    return status;
  }

  if (output_quantized != nullptr) {
    // TODO(ddavis-2015): account for optional bias tensor
    const float scale = tensors[kOutputTensor].params.scale;
    const int zero_point = tensors[kOutputTensor].params.zero_point;
    Dequantize(output_quantized, output_length, scale, zero_point, output_data);
    MicroPrintf("Dequantize: scale %f zero_point %d length %d", (double)scale,
                zero_point, output_length);
  }
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }

  return kTfLiteOk;
}

template <typename CTF = void, typename CTB = void>
TfLiteStatus TestTransposeConvFloat(
    int* input_dims_data, const float* input_data, int* filter_dims_data,
    const float* filter_data, int* bias_dims_data, const float* bias_data,
    int* output_dims_data, const float* expected_output_data,
    TfLiteConvParams* conv_params, float* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  // TODO(ddavis-2015): account for optional bias tensor

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(filter_data, filter_dims),
      CreateTensor(input_data, input_dims),
      CreateTensor(bias_data, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  const int output_dims_count = ElementCount(*output_dims);
  return ValidateTransposeConvGoldens<float>(
      tensors, tensors_size, expected_output_data, output_dims_count,
      output_data, nullptr, conv_params, kTolerance
#ifdef USE_TFLM_COMPRESSION
      ,
      comp_info
#endif  // USE_TFLM_COMPRESSION
  );
}

template <typename TBIAS, typename TIO>
TfLiteStatus TestTransposeConvQuantized(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_quantized, int* bias_dims_data,
    const float* bias_data, TBIAS* bias_quantized, int* output_dims_data,
    const float* expected_output_data, float* output_data,
    TIO* output_quantized, float output_scale, int output_zero_point,
    TfLiteConvParams* conv_params) {
  // TODO(ddavis-2015): account for optional bias tensor

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  int filter_zero_points[5];
  float filter_scales[std::extent<decltype(filter_zero_points)>::value];
  TfLiteAffineQuantization filter_quant;
  TF_LITE_MICRO_EXPECT_LE(static_cast<size_t>(filter_dims->data[0]),
                          std::extent<decltype(filter_zero_points)>::value - 1);
  TF_LITE_MICRO_CHECK_FAIL();
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  MicroPrintf(
      "input scale %f filter scale %f filter zero_point %d filter size %d %d"
      " filter qp %p %p filter data %f filter data quantized %d",
      (double)input_scale, (double)filter_quant.scale->data[0],
      filter_quant.zero_point->data[0], filter_quant.scale->size,
      filter_quant.zero_point->size, &filter_quant,
      filter_tensor.quantization.params, (double)filter_data[0],
      filter_quantized[0]);

  int bias_zero_points[std::extent<decltype(filter_zero_points)>::value];
  float bias_scales[std::extent<decltype(filter_scales)>::value];
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor bias_tensor = {};
  // TODO(ddavis-2015): cleanup
  if (filter_quant.scale->size > 0) {
    bias_tensor = CreatePerChannelQuantizedBiasTensor(
        bias_data, bias_quantized, bias_dims, input_scale,
        filter_quant.scale->data, bias_scales, bias_zero_points, &bias_quant,
        0 /* quantized dimension */);
    int64_t bias_data0 = bias_quantized[0];
    MicroPrintf(
        "bias scale %f bias zero_point %d bias size %d %d bias qp %p %p"
        " bias data %f bias data quantized %lld",
        (double)bias_quant.scale->data[0], bias_quant.zero_point->data[0],
        bias_quant.scale->size, bias_quant.zero_point->size, &bias_quant,
        bias_tensor.quantization.params, (double)bias_data[0], bias_data0);
  } else {
    bias_tensor =
        CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                  input_scale, filter_quant.scale->data[0]);

    int64_t bias_data0 = bias_quantized[0];
    MicroPrintf(
        "bias scale %f bias zero_point %d bias qp %p bias data %f bias data "
        "quantized %lld",
        (double)bias_tensor.params.scale, bias_tensor.params.zero_point,
        bias_tensor.quantization.params, (double)bias_data[0], bias_data0);
  }

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      bias_tensor,
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  // TODO(ddavis-2015): investigate why the tolerance differs from the TfLite
  // tests which use 1e-5
  //
  // Tolerance is slightly looser for 8x16 compared with float, since quant
  // error is more pronounced on the finer-grained 16-bit output.
  constexpr float tolerance = std::is_same<TIO, int8_t>::value ? 2.0f : 4.0f;
  const int output_dims_count = ElementCount(*output_dims);
  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_data, output_dims_count,
      output_data, output_quantized, conv_params, tolerance);
}

#ifdef USE_TFLM_COMPRESSION

template <typename TIO, typename CTB>
TfLiteStatus TestTransposeConvQuantizedCompressed(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* output_dims_data,
    const float* expected_output_data, float* output_data,
    TIO* output_quantized, float output_scale, int output_zero_point,
    TfLiteConvParams* conv_params,
    const TestCompressionQuantizedInfo<CTB>* comp_info) {
  // TODO(ddavis-2015): account for optional bias tensor
  MicroPrintf("%s", __PRETTY_FUNCTION__);

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(comp_info->filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(comp_info->bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteFloatArray* filter_scales =
      FloatArrayFromFloats(comp_info->filter_scales);
  TfLiteIntArray* filter_zero_points =
      IntArrayFromInts(comp_info->filter_zero_points);
  TfLiteFloatArray* bias_scales = FloatArrayFromFloats(comp_info->bias_scales);
  TfLiteIntArray* bias_zero_points =
      IntArrayFromInts(comp_info->bias_zero_points);

  size_t quantized_axis;

  TfLiteAffineQuantization filter_quant_params;
  quantized_axis = comp_info->use_filter_alt_axis ? 3 : 0;
  TfLiteTensor filter_tensor = CreatePerChannelQuantizedTensor(
      comp_info->filter_compressed, filter_dims, filter_scales,
      filter_zero_points, &filter_quant_params, quantized_axis, false,
      kTfLiteInt8);
  SymmetricPerChannelQuantize(
      comp_info->filter_data, comp_info->filter_value_table,
      ElementCount(*filter_dims), filter_dims->data[quantized_axis],
      filter_scales->data);

  TfLiteAffineQuantization bias_quant_params;
  quantized_axis = comp_info->use_bias_alt_axis ? 3 : 0;
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      comp_info->bias_compressed, bias_dims, input_scale, filter_scales,
      bias_scales, bias_zero_points, &bias_quant_params, quantized_axis, false,
      typeToTfLiteType<CTB>());
  SymmetricPerChannelQuantize(comp_info->bias_data, comp_info->bias_value_table,
                              ElementCount(*bias_dims),
                              bias_dims->data[quantized_axis],
                              bias_scales->data);
  for (int i = 0; i < bias_scales->size; i++) {
    int64_t bias_data0 = comp_info->bias_value_table[i];
    MicroPrintf(
        "bias scale %f bias zero_point %d bias size %d %d bias qp %p %p"
        " bias data %f bias data quantized %lld",
        (double)bias_quant_params.scale->data[i],
        bias_quant_params.zero_point->data[i], bias_quant_params.scale->size,
        bias_quant_params.zero_point->size, &bias_quant_params,
        bias_tensor.quantization.params, (double)comp_info->bias_data[i],
        bias_data0);
  }

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      bias_tensor,
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  // TODO(ddavis-2015): why is int8 tolerance so large?
  //
  // Tolerance is slightly looser for 8x16 compared with float, since quant
  // error is more pronounced on the finer-grained 16-bit output.
  constexpr float tolerance = std::is_same<TIO, int8_t>::value ? 2.0f : 0.19f;
  const int output_dims_count = ElementCount(*output_dims);
  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_data, output_dims_count,
      output_data, output_quantized, conv_params, tolerance, comp_info);
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// TODO(ddavis-2015): add tests with no bias tensor

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
  float output_data[tflite::testing::kOutputElements];

  // compressed filter data for kBinQuant scheme
  constexpr uint8_t kBinQuantFilterData[] = {
      0x00, 0x44, 0x32, 0x14, 0xC7, 0x42, 0x54, 0xB6, 0x35, 0xCF, 0x84, 0x40};
  constexpr int kBinQuantFilterBitWidth = 5;
  // compressed bias data for kBinQuant scheme
  constexpr uint8_t kBinQuantBiasData[] = {0x00};
  constexpr int kBinQuantBiasBitWidth = 1;

  tflite::testing::TestCompressionInfo<const float, const float> comp_info = {};
  comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  comp_info.filter_value_table = tflite::testing::kFilterData;
  comp_info.filter_value_table_stride =
      std::extent<decltype(tflite::testing::kFilterData)>::value;
  comp_info.filter_bit_width = kBinQuantFilterBitWidth;
  comp_info.bias_value_table = tflite::testing::kBiasData;
  comp_info.bias_value_table_stride =
      std::extent<decltype(tflite::testing::kBiasData)>::value;
  comp_info.bias_bit_width = kBinQuantBiasBitWidth;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          tflite::testing::kFilterShape,
          reinterpret_cast<const float*>(kBinQuantFilterData),
          tflite::testing::kBiasShape,
          reinterpret_cast<const float*>(kBinQuantBiasData),
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          &tflite::testing::common_conv_params, output_data, &comp_info));
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

#ifdef notdef
// TODO(ddavis-2015): remove
TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannel) {
  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  auto mm = std::minmax_element(std::begin(tflite::testing::kInputData),
                                std::end(tflite::testing::kInputData));
  const float input_scale =
      tflite::testing::ScaleFromMinMax<int8_t>(*mm.first, *mm.second);
  const int input_zero_point =
      tflite::testing::ZeroPointFromMinMax<int8_t>(*mm.first, *mm.second);
  mm = std::minmax_element(std::begin(tflite::testing::kGoldenData),
                           std::end(tflite::testing::kGoldenData));
  const float output_scale =
      tflite::testing::ScaleFromMinMax<int8_t>(*mm.first, *mm.second);
  const int output_zero_point =
      tflite::testing::ZeroPointFromMinMax<int8_t>(*mm.first, *mm.second);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}
#endif

TF_LITE_MICRO_TEST(SimpleBiasTestQuantizedPerChannelSingleChannel) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannelSingleChannel
  const float input_scale = 16.0f / 255.0f;
  const float output_scale = 2.0f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  int8_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ1];
  int32_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int8_t output_quantized[tflite::testing::kOutputElementsQ1];
  float output_data[tflite::testing::kOutputElementsQ1];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShapeQ1, tflite::testing::kFilterDataQ1,
          filter_quantized, tflite::testing::kBiasShapeQ1,
          tflite::testing::kBiasDataQ1, bias_quantized,
          tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelSingleChannelCompressed) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannelSingleChannel
  const float input_scale = 16.0f / 255.0f;
  const float output_scale = 2.0f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  constexpr float kFilterScales[] = {1, 9.0f / 127.0f};
  constexpr int kFilterZeroPoints[] = {1, 0};
  // all values will be computed
  float kBiasScales[std::extent<decltype(kFilterScales)>::value] = {};
  // all values will be computed
  int kBiasZeroPoints[std::extent<decltype(kFilterZeroPoints)>::value] = {};

  int8_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ1];
  int32_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int8_t output_quantized[tflite::testing::kOutputElementsQ1];
  float output_data[tflite::testing::kOutputElementsQ1];

  // compressed filter data for kBinQuant scheme
  constexpr uint8_t kBinQuantFilterData[] = {0x01, 0x23, 0x45, 0x67, 0x80};
  constexpr int kBinQuantFilterBitWidth = 4;
  // compressed bias data for kBinQuant scheme
  constexpr uint8_t kBinQuantBiasData[] = {0x00};
  constexpr int kBinQuantBiasBitWidth = 1;

  tflite::testing::TestCompressionQuantizedInfo<int32_t> comp_info = {};
  comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  comp_info.filter_value_table = filter_quantized;
  comp_info.filter_value_table_stride =
      std::extent<decltype(tflite::testing::kFilterDataQ1)>::value;
  comp_info.filter_bit_width = kBinQuantFilterBitWidth;
  comp_info.filter_compressed = kBinQuantFilterData;
  comp_info.filter_data = tflite::testing::kFilterDataQ1;
  comp_info.filter_dims_data = tflite::testing::kFilterShapeQ1;
  comp_info.filter_scales = kFilterScales;
  comp_info.filter_zero_points = kFilterZeroPoints;
  comp_info.bias_value_table = bias_quantized;
  comp_info.bias_value_table_stride =
      std::extent<decltype(tflite::testing::kBiasDataQ1)>::value;
  comp_info.bias_bit_width = kBinQuantBiasBitWidth;
  comp_info.bias_compressed = kBinQuantBiasData;
  comp_info.bias_data = tflite::testing::kBiasDataQ1;
  comp_info.bias_dims_data = tflite::testing::kBiasShapeQ1;
  comp_info.bias_scales = kBiasScales;
  comp_info.bias_zero_points = kBiasZeroPoints;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, &comp_info));
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleBiasTestQuantizedPerChannelBias16MultiChannel) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
  const float input_scale = 4.0f / 127.0f;
  const float output_scale = 128.0f / 65536.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElementsQ2];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ2];
  int16_t bias_quantized[tflite::testing::kBiasElementsQ2];
  int16_t output_quantized[tflite::testing::kOutputElementsQ2];
  float output_data[tflite::testing::kOutputElementsQ2];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShapeQ2, tflite::testing::kInputDataQ2,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShapeQ2, tflite::testing::kFilterDataQ2,
          filter_quantized, tflite::testing::kBiasShapeQ2,
          tflite::testing::kBiasDataQ2, bias_quantized,
          tflite::testing::kOutputShapeQ2, tflite::testing::kGoldenDataQ2,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(
    SimpleBiasTestQuantizedPerChannelBias16MultiChannelCompressed) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
  const float input_scale = 4.0f / 127.0f;
  const float output_scale = 128.0f / 65536.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  constexpr int kNumChannels = 2;

  constexpr float kFilterScales[] = {kNumChannels, 7.0f / 127.0f,
                                     8.0f / 127.0f};
  constexpr int kFilterZeroPoints[] = {kNumChannels, 0, 0};
  // all values will be computed
  float kBiasScales[std::extent<decltype(kFilterScales)>::value] = {};
  // all values will be computed
  int kBiasZeroPoints[std::extent<decltype(kFilterZeroPoints)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ2];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ2];
  int16_t bias_quantized[tflite::testing::kBiasElementsQ2];
  int16_t output_quantized[tflite::testing::kOutputElementsQ2];
  float output_data[tflite::testing::kOutputElementsQ2];

  // compressed filter data for kBinQuant scheme
  constexpr uint8_t kBinQuantFilterData[] = {0x05, 0x34, 0xE5,
                                             0xDE, 0x54, 0xC1};
  constexpr float kBinQuantFilterValueTable[] = {1, 2, 3, 4, 5, 6, 0, 0,
                                                 1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int kBinQuantFilterBitWidth = 3;
  // compressed bias data for kBinQuant scheme
  constexpr uint8_t kBinQuantBiasData[] = {0x00};
  constexpr int kBinQuantBiasBitWidth = 1;

  tflite::testing::TestCompressionQuantizedInfo<int16_t> comp_info = {};
  comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  comp_info.filter_value_table = filter_quantized;
  comp_info.filter_value_table_stride =
      std::extent<decltype(kBinQuantFilterValueTable)>::value / kNumChannels;
  comp_info.filter_bit_width = kBinQuantFilterBitWidth;
  comp_info.filter_compressed = kBinQuantFilterData;
  comp_info.filter_data = kBinQuantFilterValueTable;
  comp_info.filter_dims_data = tflite::testing::kFilterShapeQ2;
  comp_info.filter_scales = kFilterScales;
  comp_info.filter_zero_points = kFilterZeroPoints;
  comp_info.bias_value_table = bias_quantized;
  comp_info.bias_value_table_stride =
      std::extent<decltype(tflite::testing::kBiasDataQ2)>::value / kNumChannels;
  comp_info.bias_bit_width = kBinQuantBiasBitWidth;
  comp_info.bias_compressed = kBinQuantBiasData;
  comp_info.bias_data = tflite::testing::kBiasDataQ2;
  comp_info.bias_dims_data = tflite::testing::kBiasShapeQ2;
  comp_info.bias_scales = kBiasScales;
  comp_info.bias_zero_points = kBiasZeroPoints;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShapeQ2, tflite::testing::kInputDataQ2,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ2, tflite::testing::kGoldenDataQ2,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, &comp_info));
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleBiasTestQuantizedPerChannelBias64MultiChannel) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
  const float input_scale = 4.0f / 127.0f;
  const float output_scale = 128.0f / 65536.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElementsQ2];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ2];
  int64_t bias_quantized[tflite::testing::kBiasElementsQ2];
  int16_t output_quantized[tflite::testing::kOutputElementsQ2];
  float output_data[tflite::testing::kOutputElementsQ2];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShapeQ2, tflite::testing::kInputDataQ2,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShapeQ2, tflite::testing::kFilterDataQ2,
          filter_quantized, tflite::testing::kBiasShapeQ2,
          tflite::testing::kBiasDataQ2, bias_quantized,
          tflite::testing::kOutputShapeQ2, tflite::testing::kGoldenDataQ2,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(
    SimpleBiasTestQuantizedPerChannelBias64MultiChannelCompressed) {
  // data from TfLite test: SimpleBiasTestQuantizedPerChannel16x8Bias64
  const float input_scale = 4.0f / 127.0f;
  const float output_scale = 128.0f / 65536.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  constexpr int kNumChannels = 2;

  constexpr float kFilterScales[] = {kNumChannels, 7.0f / 127.0f,
                                     8.0f / 127.0f};
  constexpr int kFilterZeroPoints[] = {kNumChannels, 0, 0};
  // all values will be computed
  float kBiasScales[std::extent<decltype(kFilterScales)>::value] = {};
  // all values will be computed
  int kBiasZeroPoints[std::extent<decltype(kFilterZeroPoints)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ2];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ2];
  int64_t bias_quantized[tflite::testing::kBiasElementsQ2];
  int16_t output_quantized[tflite::testing::kOutputElementsQ2];
  float output_data[tflite::testing::kOutputElementsQ2];

  // compressed filter data for kBinQuant scheme
  constexpr uint8_t kBinQuantFilterData[] = {0x05, 0x34, 0xE5,
                                             0xDE, 0x54, 0xC1};
  constexpr float kBinQuantFilterValueTable[] = {1, 2, 3, 4, 5, 6, 0, 0,
                                                 1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int kBinQuantFilterBitWidth = 3;
  // compressed bias data for kBinQuant scheme
  constexpr uint8_t kBinQuantBiasData[] = {0x00};
  constexpr int kBinQuantBiasBitWidth = 2;

  tflite::testing::TestCompressionQuantizedInfo<int64_t> comp_info = {};
  comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  comp_info.filter_value_table = filter_quantized;
  comp_info.filter_value_table_stride =
      std::extent<decltype(kBinQuantFilterValueTable)>::value / kNumChannels;
  comp_info.filter_bit_width = kBinQuantFilterBitWidth;
  comp_info.filter_compressed = kBinQuantFilterData;
  comp_info.filter_data = kBinQuantFilterValueTable;
  comp_info.filter_dims_data = tflite::testing::kFilterShapeQ2;
  comp_info.filter_scales = kFilterScales;
  comp_info.filter_zero_points = kFilterZeroPoints;
  comp_info.bias_value_table = bias_quantized;
  comp_info.bias_value_table_stride =
      std::extent<decltype(tflite::testing::kBiasDataQ2)>::value / kNumChannels;
  comp_info.bias_bit_width = kBinQuantBiasBitWidth;
  comp_info.bias_compressed = kBinQuantBiasData;
  comp_info.bias_data = tflite::testing::kBiasDataQ2;
  comp_info.bias_dims_data = tflite::testing::kBiasShapeQ2;
  comp_info.bias_scales = kBiasScales;
  comp_info.bias_zero_points = kBiasZeroPoints;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShapeQ2, tflite::testing::kInputDataQ2,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ2, tflite::testing::kGoldenDataQ2,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, &comp_info));
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestQuantized16x8PerChannelSingleChannel) {
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int64_t bias_quantized[tflite::testing::kBiasElements];
  int16_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TEST(
    SimpleTestQuantized16x8PerChannelWithInt16BiasSingleChannel) {
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int16_t bias_quantized[tflite::testing::kBiasElements];
  int16_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TEST(InputOutputDifferentTypeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
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
      kTfLiteError,
      tflite::testing::InvokeTransposeConv(
          tensors, tensors_size, &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TEST(HybridModeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);

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
      kTfLiteError,
      tflite::testing::InvokeTransposeConv(
          tensors, tensors_size, &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TESTS_END
