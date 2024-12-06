
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

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Index of the output tensor in context->tensors, specific to
// DepthwiseConv.
constexpr int kOutputTensorIndex = 3;

constexpr int kMaxFilterChannels = 64;
constexpr int kMaxBiasChannels = 64;

#ifdef USE_TFLM_COMPRESSION

constexpr size_t kDepthwiseConvMaxTensors = 4;
constexpr size_t kDepthwiseConvMaxInputTensors = 3;

// Common inputs and outputs (quantized multi channel).
// data from TfLite test:
// PerChannelQuantizedDepthwiseConvolutionOpTest SimpleTestMixedOutputShift
static int kInputShapeQ1[] = {4, 1, 2, 3, 2};
static constexpr float kInputDataQ1[] = {
    // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
    3,  2,   // batch = 0, y = 0, x = 0
    1,  -1,  // batch = 0, y = 0, x = 1
    -2, -3,  // batch = 0, y = 0, x = 2
    4,  3,   // batch = 0, y = 1, x = 0
    2,  -2,  // batch = 0, y = 1, x = 1
    -3, -4,  // batch = 0, y = 1, x = 2
};
constexpr size_t kInputElementsQ1 = std::extent<decltype(kInputDataQ1)>::value;

constexpr int kNumChannelsQ1 = 4;
static int kFilterShapeQ1[] = {4, 1, 2, 2, 4};
static constexpr float kFilterDataQ1[] = {
    // This is a compact value table.  Original data is:
    // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
    // depth multiplier = 2
    // 1, 2, 3, 4,   y = 0, x = 0
    // 3, 4, 5, 6,   y = 0, x = 1
    // 7, 8, 5, 6,   y = 1, x = 0
    // 3, 4, 1, 2,   y = 1, x = 1
    1, 3, 7, 8, 2, 4, 1, 3, 5, 2, 4, 6,
};
constexpr size_t kFilterElementsQ1 =
    std::extent<decltype(kFilterDataQ1)>::value;

static int kBiasShapeQ1[] = {1, 4};
static constexpr float kBiasDataQ1[] = {3, -2, 4, 6};
constexpr size_t kBiasElementsQ1 = std::extent<decltype(kBiasDataQ1)>::value;

static int kOutputShapeQ1[] = {4, 1, 1, 2, 4};
static constexpr float kGoldenDataQ1[] = {43, 48, 21, 22, 3, -4, -30, -36};
constexpr int kOutputElementsQ1 = std::extent<decltype(kGoldenDataQ1)>::value;

// compressed filter data for kBinQuant scheme, matches kFilterDataQ1
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantFilterDataQ1[] = {0x15, 0x6A, 0x8A,
                                                         0x60};
constexpr int kBinQuantFilterBitWidthQ1 = 2;
// compressed bias data for kBinQuant scheme, matches kBiasDataQ1
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantBiasDataQ1[] = {0x00};
constexpr int kBinQuantBiasBitWidthQ1 = 1;

#endif  // USE_TFLM_COMPRESSION

// Creates a DepthwiseConv opeerator, calls it with the provided input tensors
// and some defaults parameters, and compares the output with
// expected_output_data.
//
// The tensors parameter contains both the input tensors as well as a
// preallocated output tensor into which the output is stored.
template <typename T, typename TF = void, typename TB = void>
TfLiteStatus ValidateDepthwiseConvGoldens(
    const T* expected_output_data, int output_length,
    TfLiteDepthwiseConvParams* conv_params, float tolerance, int tensors_size,
    TfLiteTensor* tensors
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<TF>* filter_comp_info = nullptr,
    const TestCompressionInfo<TB>* bias_comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
#ifdef USE_TFLM_COMPRESSION

  TestCompressedList<kDepthwiseConvMaxInputTensors> tcl;
  if (filter_comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(
        tcl.AddInput(*filter_comp_info, tensors[kDepthwiseConvWeightsTensor],
                     kDepthwiseConvWeightsTensor),
        kTfLiteOk);
    TF_LITE_MICRO_CHECK_FAIL();
  }
  if (bias_comp_info != nullptr) {
    TF_LITE_MICRO_EXPECT_EQ(
        tcl.AddInput(*bias_comp_info, tensors[kDepthwiseConvBiasTensor],
                     kDepthwiseConvBiasTensor),
        kTfLiteOk);
    TF_LITE_MICRO_CHECK_FAIL();
  }
  const CompressedTensorList* comp_list_p = tcl.GetCompressedTensorList();

#endif  // USE_TFLM_COMPRESSION

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_DEPTHWISE_CONV_2D();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, reinterpret_cast<void*>(conv_params)
#ifdef USE_TFLM_COMPRESSION
                                                ,
                             nullptr, comp_list_p
#endif  // USE_TFLM_COMPRESSION
  );

  int input_depth = tensors[0].dims->data[3];
  int output_depth = tensors[1].dims->data[3];
  int depth_mul = output_depth / input_depth;

  conv_params->padding = kTfLitePaddingValid;
  conv_params->depth_multiplier = depth_mul;

  const char* init_data = reinterpret_cast<const char*>(conv_params);

  // TODO(b/154240825): Use a test macro here which fails and returns.
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const T* output_data = tflite::GetTensorData<T>(&tensors[kOutputTensorIndex]);

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

template <typename T, typename BiasT>
void TestDepthwiseConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, T* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, BiasT* bias_data_quantized,
    int* output_dims_data, const float* expected_output_data,
    T* expected_output_data_quantized, T* output_data, float output_scale,
    int output_zero_point, TfLiteDepthwiseConvParams* conv_params,
    TfLiteType filter_packed_type = kTfLiteNoType) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[kMaxFilterChannels];
  float filter_scales[kMaxFilterChannels];
  int bias_zero_points[kMaxBiasChannels];
  float bias_scales[kMaxBiasChannels];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor = CreateQuantizedTensor(
      input_data, input_quantized, input_dims, input_scale, input_zero_point);
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 3 /* quantized dimension */, false,
      filter_packed_type);
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 3 /* quantized dimension */
  );
  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_data, output_dims, output_scale, input_zero_point);

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points),
                                          0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {FloatArrayFromFloats(output_scales),
                                           IntArrayFromInts(output_zero_points),
                                           0};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  Quantize(expected_output_data, expected_output_data_quantized,
           output_dims_count, output_scale, output_zero_point);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateDepthwiseConvGoldens(expected_output_data_quantized,
                                              output_dims_count, conv_params,
                                              1.0, tensors_size, tensors));
}

void TestDepthwiseConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int8_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int32_t* bias_data_quantized,
    int* output_dims_data, const float* expected_output_data,
    int8_t* expected_output_data_quantized, int8_t* output_data,
    float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams* conv_params,
    TfLiteType filter_packed_type = kTfLiteNoType) {
  return TestDepthwiseConvQuantizedPerChannel<int8_t, int32_t>(
      input_dims_data, input_data, input_quantized, input_scale,
      input_zero_point, filter_dims_data, filter_data, filter_data_quantized,
      bias_dims_data, bias_data, bias_data_quantized, output_dims_data,
      expected_output_data, expected_output_data_quantized, output_data,
      output_scale, output_zero_point, conv_params, filter_packed_type);
}

void TestDepthwiseConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int16_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int64_t* bias_data_quantized,
    int* output_dims_data, const float* expected_output_data,
    int16_t* expected_output_data_quantized, int16_t* output_data,
    float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams* conv_params,
    TfLiteType filter_packed_type = kTfLiteNoType) {
  return TestDepthwiseConvQuantizedPerChannel<int16_t, int64_t>(
      input_dims_data, input_data, input_quantized, input_scale,
      input_zero_point, filter_dims_data, filter_data, filter_data_quantized,
      bias_dims_data, bias_data, bias_data_quantized, output_dims_data,
      expected_output_data, expected_output_data_quantized, output_data,
      output_scale, output_zero_point, conv_params, filter_packed_type);
}

#ifdef USE_TFLM_COMPRESSION

template <typename TIO, typename TBIAS>
TfLiteStatus TestDepthwiseConvQuantizedCompressed(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* output_dims_data,
    const float* expected_output_data, TIO* expected_output_quantized,
    TIO* output_quantized, float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams* conv_params, const unsigned int tolerance,
    const TestCompressionQuantizedInfo<int8_t>* filter_comp_info,
    const TestCompressionQuantizedInfo<TBIAS>* bias_comp_info) {
  // TODO(b/360169306): account for optional bias tensor
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
      filter_zero_points, &filter_quant, kDepthwiseConvQuantizedDimension,
      false /* is_variable */, kTfLiteInt8);
  // Value tables are always in channel order, therefore do not use the
  // quantized dimension.
  SymmetricPerChannelQuantize(
      filter_comp_info->data, filter_comp_info->value_table,
      filter_scales->size * filter_comp_info->value_table_stride,
      filter_scales->size, filter_scales->data, 0 /* see comment above */);

  TfLiteAffineQuantization bias_quant = {};
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_comp_info->compressed, bias_dims, input_scale, filter_scales,
      bias_scales, bias_zero_points, &bias_quant,
      0 /* quantized dimension for bias tensor */, false /* is_variable */,
      typeToTfLiteType<TBIAS>());
  SymmetricPerChannelQuantize(
      bias_comp_info->data, bias_comp_info->value_table,
      bias_scales->size * bias_comp_info->value_table_stride, bias_scales->size,
      bias_scales->data);

  constexpr int tensors_size = kDepthwiseConvMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      filter_tensor,
      bias_tensor,
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  const int output_dims_count = ElementCount(*output_dims);
  Quantize(expected_output_data, expected_output_quantized, output_dims_count,
           output_scale, output_zero_point);
  return ValidateDepthwiseConvGoldens(
      expected_output_quantized, output_dims_count, conv_params, tolerance,
      tensors_size, tensors, filter_comp_info, bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

// TODO(ddavis-2015): is this still valid?
// Xtensa kernels do not support float activations., and the corresponding tests
// are disabled. As a result, helper functions that are only needed for float
// kernel tests also need to be ifdef'd out to avoid build errors due to unused
// functions.
#if !defined(XTENSA)
void TestDepthwiseConvFloat(
    int* input_dims_data, const float* input_data, int* filter_dims_data,
    const float* filter_data, int* bias_dims_data, const float* bias_data,
    const float* expected_output_data, int* output_dims_data,
    TfLiteDepthwiseConvParams* conv_params, float* output_data
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

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(filter_data, filter_dims),
      CreateTensor(bias_data, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateDepthwiseConvGoldens(expected_output_data, output_dims_count,
                               conv_params, 1e-5, tensors_size, tensors
#ifdef USE_TFLM_COMPRESSION
                               ,
                               filter_comp_info, bias_comp_info
#endif  // USE_TFLM_COMPRESSION
  );
}

#endif  // !defined(XTENSA)

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

#if !defined(XTENSA)  // TODO(b/170322965): xtensa kernels are less general than
                      // reference kernels and we ifdef out test cases that are
                      // currently known to fail.
TF_LITE_MICRO_TEST(SimpleTest) {
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  float output_data[output_dims_count];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden, output_shape, &conv_params, output_data);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestCompressed) {
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  int filter_shape[] = {4, 1, 2, 2, 4};
  // Filter values:
  // {1, 2, 3, 4, -9, 10,  -11, 12, 5, 6, 7, 8, 13, -14, 15,  -16}
  // Align the tensor data the same as a Buffer in the schema
  alignas(16) const uint8_t kBinQuantFilterData[] = {0x01, 0x23, 0xF8, 0xE9,
                                                     0x45, 0x67, 0xAD, 0xBC};
  const float kBinQuantFilterValueTable[] = {1,  2,  3,  4,  5,   6,   7,   8,
                                             10, 12, 13, 15, -16, -14, -11, -9};
  int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_values[] = {1, 2, 3, 4};
  // Align the tensor data the same as a Buffer in the schema
  alignas(16) const uint8_t kBinQuantBiasData[] = {0x1B};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = std::extent<decltype(golden)>::value;
  float output_data[output_dims_count];

  tflite::testing::TestCompressionInfo<const float> filter_comp_info = {};
  tflite::testing::TestCompressionInfo<const float> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = kBinQuantFilterValueTable;
  filter_comp_info.value_table_stride =
      std::extent<decltype(kBinQuantFilterValueTable)>::value;
  filter_comp_info.bit_width = 4;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_values;
  bias_comp_info.value_table_stride = std::extent<decltype(bias_values)>::value;
  bias_comp_info.bit_width = 2;

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape,
      reinterpret_cast<const float*>(kBinQuantFilterData), bias_shape,
      reinterpret_cast<const float*>(kBinQuantBiasData), golden, output_shape,
      &conv_params, output_data, &filter_comp_info, &bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_values[] = {1, 2, 3, 4};
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  const float golden_relu[] = {71, 0, 99, 0, 91, 0, 127, 0};
  float output_data[output_dims_count];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActRelu;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden_relu, output_shape, &conv_params, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelDepthMultiplier1) {
  const int input_elements = 12;
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 8;
  int filter_shape[] = {4, 1, 2, 2, 2};
  const float filter_values[] = {1, 2, 3, 4, -9, 10, -11, 12};
  const int bias_elements = 2;
  int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 4;
  const float bias_values[] = {1, 2};
  const float golden[] = {
      -103,
      127,
      -128,
      127,
  };
  int output_shape[] = {4, 1, 2, 1, 2};
  const int output_dims_count = 4;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(TestQuantizedPerChannelDepthMultiplier1Relu6) {
  const int input_elements = 24;
  int input_shape[] = {4, 1, 3, 2, 4};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int filter_elements = 16;
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {0,  1, 8,   -2, -1, 2, -10, 0,
                                 -1, 3, -18, 0,  0,  4, 20,  -3};
  const int bias_elements = 4;
  int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      0, 6, 3, 0, 0, 6, 3, 0,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  int8_t output_data[output_elements];

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActRelu6;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(SimpleTestDilatedQuantizedPerChannel) {
  const int input_elements = 48;
  int input_shape[] = {4, 1, 4, 6, 2};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,   // h = 0
                                3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,   // h = 1
                                1, 2, 3, 4, 5, 6, 2, 6, 2, 4, 4, 2,   // h = 2
                                3, 2, 6, 5, 1, 4, 1, 2, 1, 4, 6, 3};  // h = 3
  const int filter_elements = 16;
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 24;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      15, 2,  88, -48, 25, 14, 72, 0,  61, -2,  56, 48,  // h = 0
      -4, 52, 12, 48,  11, 70, 63, 40, 51, -30, 41, 48   // h = 1
  };
  int output_shape[] = {4, 1, 2, 3, 4};
  int8_t output_data[output_elements];

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 3;
  conv_params.dilation_height_factor = 2;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(TestQuantizedPerChannelCompareWithFloat) {
  int input_dims[] = {4, 1, 2, 3, 2};
  const float input_data[] = {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4};
  int filter_dims[] = {4, 1, 2, 2, 4};
  const float filter_data[] = {1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 5, 6, 3, 4, 1, 2};
  int bias_dims[] = {4, 1, 1, 1, 4};
  const float bias_data[] = {3, -2, 4, 6};
  int output_dims[] = {4, 1, 1, 2, 4};
  const float golden[] = {43, 48, 18, 22, 3, -4, -28, -36};

  const int input_size = 12;
  const int filter_size = 16;
  const int output_size = 8;
  const int bias_size = 4;
  int8_t input_quantized[input_size];
  int8_t filter_quantized[filter_size];
  int32_t bias_quantized[bias_size];
  int8_t golden_quantized[output_size];
  int8_t output_data[output_size];
  float output_float[output_size];

  const float input_scale = 0.5;
  const float output_scale = 1.0;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_dims, input_data, input_quantized, input_scale, input_zero_point,
      filter_dims, filter_data, filter_quantized, bias_dims, bias_data,
      bias_quantized, output_dims, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);

  tflite::testing::TestDepthwiseConvFloat(
      input_dims, input_data, filter_dims, filter_data, bias_dims, bias_data,
      golden, output_dims, &conv_params, output_float);
}

TF_LITE_MICRO_TEST(PerChannelBroadcastQuantizationParams) {
  const float input_scale = 1.0f;
  const float filter_scale = 1.0f;
  const float output_scale = 1.0f;

  const int input_elements = 12;
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  int8_t output_data[output_dims_count];

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  // Create per-layer quantized int8_t input tensor.
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, 0);
  int input_zero_points[2] = {1, 0};
  float input_scales[2] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-layer quantized int8_t filter tensor.
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0);
  int filter_zero_points[2] = {1, 0};
  float filter_scales[2] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-layer quantized int32_t bias tensor.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  int bias_zero_points[2] = {1, 0};
  float bias_scales[2] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-layer quantized int8_t output tensor.
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, 0);
  int output_zero_points[2] = {1, 0};
  float output_scales[2] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  tflite::Quantize(golden, golden_quantized, output_dims_count, output_scale,
                   0);

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::ValidateDepthwiseConvGoldens(
                     golden_quantized, output_dims_count, &conv_params, 1e-5,
                     tensors_size, tensors));
}

#endif  // !defined(XTENSA)

TF_LITE_MICRO_TEST(FilterDimsNotMatchingAffineQuantization) {
  int input_shape[] = {4, 1, 2, 3, 2};
  const float input_data[] = {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4};
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_data[] = {1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 5, 6, 3, 4, 1, 2};
  int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_data[] = {3, -2, 4, 6};
  int output_shape[] = {4, 1, 1, 2, 4};

  const int input_size = 12;
  const int filter_size = 16;
  const int output_size = 8;
  const int bias_size = 4;
  int8_t input_quantized[input_size];
  int8_t filter_quantized[filter_size];
  int32_t bias_quantized[bias_size];
  int8_t golden_quantized[output_size] = {};
  int zero_points[bias_size + 1];
  float scales[bias_size + 1];
  int8_t output_data[output_size];

  const float input_scale = 0.5;
  const float output_scale = 1.0;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_quantized, input_dims, input_scale, input_zero_point);
  TfLiteTensor filter_tensor =
      tflite::testing::CreateSymmetricPerChannelQuantizedTensor(
          filter_data, filter_quantized, filter_dims, filter_scales,
          filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  TfLiteTensor bias_tensor =
      tflite::testing::CreatePerChannelQuantizedBiasTensor(
          bias_data, bias_quantized, bias_dims, input_scale, &filter_scales[1],
          scales, zero_points, &bias_quant, 0);
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, output_zero_point);

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  // Set filter quant to mismatched dimension.
  TfLiteAffineQuantization* quant = reinterpret_cast<TfLiteAffineQuantization*>(
      filter_tensor.quantization.params);
  quant->scale->size = 2;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::ValidateDepthwiseConvGoldens(
                              golden_quantized, output_size, &conv_params, 1e-5,
                              tensors_size, tensors));

  // Set scale back to correct dimension, and make zero point array too short.
  quant->scale->size = filter_shape[0];
  quant->zero_point->size = 2;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::ValidateDepthwiseConvGoldens(
                              golden_quantized, output_size, &conv_params, 1e-5,
                              tensors_size, tensors));
}

TF_LITE_MICRO_TEST(Int8Input32x4Filter32x4ShouldMatchGolden) {
  const int input_elements = 32 * 4;
  const int filter_elements = 32 * 4;
  const int bias_elements = 32;
  const int output_elements = 32;
  int input_shape[] = {4, 1, 4, 1, 32};
  int filter_shape[] = {4, 1, 4, 1, 32};
  int bias_shape[] = {1, 32};
  int output_shape[] = {4, 1, 1, 1, 32};
  const float input_values[] = {
      11.0589, 10.8824, 11.1766, 11.5295, 10.8236, 9.5295, 9.5295, 10.0001,
      11.2354, 10.8824, 9.1765,  9.0589,  9.6471,  8.9412, 7.9412, 9.0001,
      9.3530,  7.5295,  9.2354,  9.5883,  7.5883,  8.1765, 7.5883, 9.2942,
      9.1177,  8.5883,  8.2354,  8.6471,  8.0589,  8.0001, 7.4118, 7.3530,
      11.0001, 11.1177, 11.0589, 11.2354, 10.5883, 9.2942, 9.2942, 10.1177,
      11.2354, 10.8824, 8.9412,  8.8236,  9.2354,  8.8824, 7.0001, 9.1177,
      9.5883,  8.2354,  9.1765,  9.5295,  7.4118,  8.5883, 8.1177, 9.1765,
      9.0001,  9.0589,  8.9412,  8.2942,  7.8824,  8.4118, 7.2942, 7.2354,
      10.4118, 10.8824, 11.1177, 11.0001, 10.0001, 9.7060, 9.7648, 10.1766,
      11.1766, 10.6471, 8.6471,  8.5295,  9.5295,  9.0001, 7.0001, 9.4118,
      9.8236,  8.0001,  9.2354,  9.5883,  7.5295,  9.0001, 8.5295, 9.0589,
      8.9412,  9.1177,  8.9412,  8.0001,  8.0589,  8.8824, 7.0589, 7.3530,
      11.3530, 11.0589, 10.7060, 10.7648, 9.9413,  9.1177, 9.1177, 9.7648,
      10.7060, 10.2354, 8.5883,  8.8236,  9.7648,  9.2942, 7.5295, 9.2354,
      9.7060,  8.1177,  9.2942,  9.5883,  7.7648,  9.6471, 9.1177, 9.4707,
      9.3530,  8.8236,  8.5295,  8.0589,  8.6471,  9.5883, 7.4118, 7.5883};
  const float filter_values[] = {
      -0.1617, -0.1948, 0.1419,  -0.2311, -0.0891, 0.1551,  0.0033,  0.3037,
      -0.1683, 0.1353,  0.1518,  -0.1683, -0.1386, 0.1452,  0.1816,  0.1716,
      -0.1948, 0.2080,  0.2245,  -0.1981, -0.2410, 0.1849,  0.1981,  0.1584,
      0.2509,  0.1783,  -0.2146, -0.1518, 0.2080,  -0.2872, 0.2014,  0.2476,
      -0.4126, -0.0561, -0.3235, -0.0594, -0.0957, 0.2014,  -0.1056, 0.1386,
      -0.2542, -0.1617, 0.1287,  -0.1816, -0.0363, 0.1419,  -0.0594, 0.2344,
      -0.0099, 0.4192,  0.1287,  -0.2311, -0.2212, -0.0528, -0.2080, 0.1816,
      -0.1452, 0.1221,  0.1254,  -0.1056, -0.0759, 0.1221,  0.1023,  0.1485,
      0.2707,  0.1716,  -0.1882, -0.1783, 0.1650,  -0.2740, 0.1915,  0.2080,
      -0.2971, -0.2575, -0.3169, 0.0198,  -0.0231, 0.2410,  -0.0429, 0.0660,
      -0.1816, 0.1981,  0.2014,  -0.1386, -0.1915, 0.1716,  0.1320,  0.1419,
      0.1320,  0.1353,  -0.1386, -0.1716, 0.1320,  -0.1650, 0.1386,  0.0825,
      -0.1419, -0.1023, 0.1783,  0.0462,  0.2047,  -0.2179, -0.1518, -0.1551,
      0.1518,  0.3334,  0.3103,  -0.2047, -0.2047, -0.0957, -0.1650, 0.1221,
      0.0990,  0.1353,  -0.1617, -0.1485, 0.1650,  -0.1816, 0.1518,  0.1254,
      -0.0363, -0.1254, 0.1386,  0.0429,  0.2113,  -0.2839, -0.1056, -0.2278};
  const float bias_values[] = {
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
  const float golden[] = {
      -5.1194, -2.0075, -2.1751, -4.7958, 1.7073,  -1.2963, -0.4641, 5.0416,
      -6.4424, 0.3836,  2.4684,  -4.7643, -3.8913, 3.8382,  -0.5164, 5.4304,
      -2.7400, 7.7016,  3.6115,  -6.8545, -3.6290, 0.8509,  2.3247,  5.6117,
      1.8215,  2.7645,  -0.7032, -3.2156, 3.9689,  -5.4583, 2.4346,  1.7731};

  // Quantization Parameters.  All scales except output are 1.0, and all zero
  // points are 0. This direct-maps the values to floating point and makes it
  // easy to reson about them.
  const float input_scale = 0.058824;
  const float filter_scale = 0.003301;
  const float output_scale = 0.092596;
  const int input_zero_point = -128;
  const int output_zero_point = 0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  // Create per-tensor quantized int8_t input tensor.
  int8_t input_quantized[input_elements];
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, input_zero_point);

  // Set zero point and scale arrays with a single element for each.
  int input_zero_points[] = {1, input_zero_point};
  float input_scales[] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-tensor quantized int8_t filter tensor.
  int8_t filter_quantized[filter_elements];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0);

  // Set zero point and scale arrays with a single element for each.
  int filter_zero_points[] = {1, 0};
  float filter_scales[] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-tensor quantized int32_t bias tensor.
  int32_t bias_quantized[bias_elements];
  // See https://www.tensorflow.org/lite/performance/quantization_spec for a
  // detailed explanation of why bias scale is input_scale * filter_scale.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  // Set zero point and scale arrays with a single element for each.
  int bias_zero_points[] = {1, 0};
  float bias_scales[] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-tensor quantized int8_t output tensor.
  int8_t output_quantized[output_elements];
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point);

  // Set zero point and scale arrays with a single element for each.
  int output_zero_points[] = {1, output_zero_point};
  float output_scales[] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  int8_t golden_quantized[output_elements];
  tflite::Quantize(golden, golden_quantized, output_elements, output_scale, 0);

  // Errors due to quantization should not exceed 1.
  constexpr int kQuantizationTolerance = 1;

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;
  tflite::testing::ValidateDepthwiseConvGoldens(
      golden_quantized, output_elements, &conv_params, kQuantizationTolerance,
      kTensorsSize, tensors);
}

TF_LITE_MICRO_TEST(Int8Input32x1Filter32x1ShouldMatchGolden) {
  const int input_elements = 32 * 1;
  const int filter_elements = 32 * 1;
  const int bias_elements = 32;
  const int output_elements = 32;
  int input_shape[] = {4, 1, 1, 1, 32};
  int filter_shape[] = {4, 1, 1, 1, 32};
  int bias_shape[] = {1, 32};
  int output_shape[] = {4, 1, 1, 1, 32};
  const float input_values[] = {
      11.0589, 10.8824, 11.1766, 11.5295, 10.8236, 9.5295, 9.5295, 10.0001,
      11.2354, 10.8824, 9.1765,  9.0589,  9.6471,  8.9412, 7.9412, 9.0001,
      9.3530,  7.5295,  9.2354,  9.5883,  7.5883,  8.1765, 7.5883, 9.2942,
      9.3530,  8.8236,  8.5295,  8.0589,  8.6471,  9.5883, 7.4118, 7.5883};
  const float filter_values[] = {
      -0.1419, -0.1023, 0.1783,  0.0462,  0.2047,  -0.2179, -0.1518, -0.1551,
      0.1518,  0.3334,  0.3103,  -0.2047, -0.2047, -0.0957, -0.1650, 0.1221,
      0.0990,  0.1353,  -0.1617, -0.1485, 0.1650,  -0.1816, 0.1518,  0.1254,
      -0.0363, -0.1254, 0.1386,  0.0429,  0.2113,  -0.2839, -0.1056, -0.2278};
  const float bias_values[] = {
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
  const float golden[] = {
      -1.5741, -1.1112, 2.0371,  0.5556,  2.2223,  -2.0371, -1.4815, -1.5741,
      1.6667,  3.6112,  2.8705,  -1.8519, -1.9445, -0.8334, -1.2963, 1.1112,
      0.9260,  1.0186,  -1.4815, -1.3889, 1.2963,  -1.4815, 1.1112,  1.2037,
      -0.3704, -1.1112, 1.2037,  0.3704,  1.8519,  -2.6853, -0.7408, -1.7593};

  // Quantization Parameters.  All scales except output are 1.0, and all zero
  // points are 0. This direct-maps the values to floating point and makes it
  // easy to reson about them.
  const float input_scale = 0.058824;
  const float filter_scale = 0.003301;
  const float output_scale = 0.092596;
  const int input_zero_point = -128;
  const int output_zero_point = 0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  // Create per-tensor quantized int8_t input tensor.
  int8_t input_quantized[input_elements];
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, input_zero_point);

  // Set zero point and scale arrays with a single element for each.
  int input_zero_points[] = {1, input_zero_point};
  float input_scales[] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-tensor quantized int8_t filter tensor.
  int8_t filter_quantized[filter_elements];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0);

  // Set zero point and scale arrays with a single element for each.
  int filter_zero_points[] = {1, 0};
  float filter_scales[] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-tensor quantized int32_t bias tensor.
  int32_t bias_quantized[bias_elements];
  // See https://www.tensorflow.org/lite/performance/quantization_spec for a
  // detailed explanation of why bias scale is input_scale * filter_scale.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  // Set zero point and scale arrays with a single element for each.
  int bias_zero_points[] = {1, 0};
  float bias_scales[] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-tensor quantized int8_t output tensor.
  int8_t output_quantized[output_elements];
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point);

  // Set zero point and scale arrays with a single element for each.
  int output_zero_points[] = {1, output_zero_point};
  float output_scales[] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  int8_t golden_quantized[output_elements];
  tflite::Quantize(golden, golden_quantized, output_elements, output_scale, 0);

  // Errors due to quantization should not exceed 1.
  constexpr int kQuantizationTolerance = 1;

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 2;
  conv_params.stride_width = 2;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::ValidateDepthwiseConvGoldens(
                              golden_quantized, output_elements, &conv_params,
                              kQuantizationTolerance, kTensorsSize, tensors));
}

// TODO(b/268384678): xtensa vision p6 kernels break
// this test, will if def till properly investigated.

// Quantizing int8-ranged filter values down to int4 doesn't always yield the
// accuracy sufficient to meet the golden values. So this test was created by
// handcrafting filter values within the int4 range, and the golden data was
// obtained by running TestDepthwiseConvQuantizedPerChannel() with int8
// quantization, and ensuring that int4 quantization yields the same outputs.
TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelInt4Filter) {
  const int input_elements = 12;
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -5, 7,  -6, 7,
                                 5, 6, 7, 4, 2,  -5, 4,  0};
  const int bias_elements = 4;
  int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      0, 26, 29, 84, 6, 46, 45, 114,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params, kTfLiteInt4);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannel) {
  const int input_elements = 12;
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelInt16InputInt8Filter) {
  const int input_elements = 12;
  int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {-547, 108, -682, 540,  -161, -539, 9,    -482,
                                -859, 84,  153,  -726, 523,  702,  -172, -936};
  const int filter_elements = 16;
  int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      4894, -9009, -16596, 10268, -2564, -7483, -6599, 4356,
  };
  int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  int16_t output_data[output_dims_count];

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int64_t bias_quantized[bias_elements];
  int16_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
}

#ifdef USE_TFLM_COMPRESSION

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelInt8Compressed) {
  // data from TfLite test:
  // PerChannelQuantizedDepthwiseConvolutionOpTest SimpleTestMixedOutputShift
  const float input_scale = 0.5f;
  const float output_scale = 0.5f;
  const int input_zero_point = -1;
  const int output_zero_point = -1;
  constexpr float filter_scales[] = {
      tflite::testing::kNumChannelsQ1, 0.1f, 2.0f, 3.0f, 0.4f,
  };
  constexpr int filter_zero_points[] = {
      tflite::testing::kNumChannelsQ1, 0, 0, 0, 0,
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

  TfLiteDepthwiseConvParams conv_params = {};
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  // tolerance of 3 is approx. 2.0f
  // TODO(ddavis-2015): why does the tolerance differ from TfLite test???
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestDepthwiseConvQuantizedCompressed(
          tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1,
          golden_quantized, output_quantized, output_scale, output_zero_point,
          &conv_params, 3, &filter_comp_info, &bias_comp_info));
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelInt16Compressed) {
  // data from TfLite test:
  // PerChannelQuantizedDepthwiseConvolutionOpTest SimpleTestMixedOutputShift
  const float input_scale =
      tflite::testing::SymmetricScaleFromMinMax<int16_t>(-4.0f, 4.0f);
  const float output_scale =
      tflite::testing::SymmetricScaleFromMinMax<int16_t>(-63.5f, 64.0f);
  const int input_zero_point = 0;
  const int output_zero_point = 0;
  constexpr float filter_scales[] = {
      tflite::testing::kNumChannelsQ1, 0.1f, 2.0f, 3.0f, 0.4f,
  };
  constexpr int filter_zero_points[] = {
      tflite::testing::kNumChannelsQ1, 0, 0, 0, 0,
  };
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kFilterElementsQ1];
  int64_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int16_t golden_quantized[tflite::testing::kOutputElementsQ1];
  int16_t output_quantized[tflite::testing::kOutputElementsQ1];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int64_t> bias_comp_info = {};

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

  TfLiteDepthwiseConvParams conv_params = {};
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;

  // tolerance of 512 is approx. 1.0f
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestDepthwiseConvQuantizedCompressed(
          tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1,
          golden_quantized, output_quantized, output_scale, output_zero_point,
          &conv_params, 512, &filter_comp_info, &bias_comp_info));
}

#endif  // USE_TFLM_COMPRESSION

TF_LITE_MICRO_TESTS_END
