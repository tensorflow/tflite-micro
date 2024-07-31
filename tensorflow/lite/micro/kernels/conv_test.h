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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

constexpr int kMaxTensors = 4;

#ifdef USE_TFLM_COMPRESSION

template <typename TFILTER, typename TBIAS>
struct TestCompressionInfo {
  TFILTER* filter_value_table;
  size_t filter_value_table_stride;
  int filter_bit_width;
  TBIAS* bias_value_table;
  size_t bias_value_table_stride;
  int bias_bit_width;
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

template <typename T>
TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, const TfLiteConvParams* conv_params,
                        TFLMRegistration registration, T* output_data
#ifdef USE_TFLM_COMPRESSION
                        ,
                        const CompressedTensorList* comp_list_p = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
  // TODO(ddavis-2015): support optional bias tensor
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

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

template <typename T, typename CTF = void, typename CTB = void>
TfLiteStatus ValidateConvGoldens(
    TfLiteTensor* tensors, int tensors_size, const T* expected_output_data,
    int output_length, const TfLiteConvParams* conv_params,
    TFLMRegistration registration, T* output_data, float tolerance = 1e-5
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr
#endif  // USE_TFLM_COMPRESSION
) {
#ifdef USE_TFLM_COMPRESSION

  CompressionTensorData* compressed_tensors[kMaxTensors] = {};
  CompressionTensorData filter_comp_data = {};
  LookupTableData filter_lut_table = {};
  filter_comp_data.data.lut_data = &filter_lut_table;
  CompressionTensorData bias_comp_data = {};
  LookupTableData bias_lut_table = {};
  bias_comp_data.data.lut_data = &bias_lut_table;
  CompressedTensorList comp_list = {compressed_tensors};
  CompressedTensorList* comp_list_p = nullptr;

  if (comp_info != nullptr) {
    if (comp_info->scheme == CompressionScheme::kBinQuant) {
      if (comp_info->filter_value_table != nullptr) {
        bool is_per_channel =
            tensors[kConvWeightsTensor].type != kTfLiteFloat32 &&
            tensors[kConvWeightsTensor].dims->data[kConvQuantizedDimension] > 1;
        compressed_tensors[kConvWeightsTensor] = &filter_comp_data;
        filter_comp_data.scheme = CompressionScheme::kBinQuant;
        filter_comp_data.data.lut_data->compressed_bit_width =
            comp_info->filter_bit_width;
        filter_comp_data.data.lut_data->value_table =
            comp_info->filter_value_table;
        filter_comp_data.data.lut_data->value_table_channel_stride =
            comp_info->filter_value_table_stride;
        filter_comp_data.data.lut_data->is_per_channel_quantized =
            is_per_channel;
        filter_comp_data.data.lut_data->use_alternate_axis = false;
      }
      if (comp_info->bias_value_table != nullptr) {
        bool is_per_channel =
            tensors[kConvBiasTensor].type != kTfLiteFloat32 &&
            tensors[kConvBiasTensor].dims->data[kConvQuantizedDimension] > 1;
        compressed_tensors[kConvBiasTensor] = &bias_comp_data;
        bias_comp_data.scheme = CompressionScheme::kBinQuant;
        bias_comp_data.data.lut_data->compressed_bit_width =
            comp_info->bias_bit_width;
        bias_comp_data.data.lut_data->value_table =
            comp_info->bias_value_table;
        bias_comp_data.data.lut_data->value_table_channel_stride =
            comp_info->bias_value_table_stride;
        bias_comp_data.data.lut_data->is_per_channel_quantized =
            is_per_channel;
        bias_comp_data.data.lut_data->use_alternate_axis = false;
      }
      comp_list_p = &comp_list;
    } else {
      return kTfLiteError;
    }
  }

#endif  // USE_TFLM_COMPRESSION

  TfLiteStatus status = InvokeConv(tensors, tensors_size, output_length,
                                   conv_params, registration, output_data
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

template <typename CTF = void, typename CTB = void>
TfLiteStatus TestConvFloat(
    int* input_dims_data, const float* input_data, int* filter_dims_data,
    const float* filter_data, int* bias_dims_data, const float* bias_data,
    int* output_dims_data, const float* expected_output_data,
    TfLiteConvParams* conv_params, TFLMRegistration registration,
    float* output_data
#ifdef USE_TFLM_COMPRESSION
    ,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr
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

  return ValidateConvGoldens(tensors, tensors_size, expected_output_data,
                             output_dims_count, conv_params, registration,
                             output_data
#ifdef USE_TFLM_COMPRESSION
                             ,
                             1e-5, comp_info
#endif  // USE_TFLM_COMPRESSION
  );
}

TfLiteStatus TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int8_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int32_t* bias_data_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TFLMRegistration registration, int8_t* output_data,
    TfLiteType tensor_weight_type = kTfLiteNoType);

TfLiteStatus TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int16_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data,
    std::int64_t* bias_data_quantized, float* bias_scales,
    int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int16_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TFLMRegistration registration, int16_t* output_data);

TfLiteStatus TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int16_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int32_t* bias_data_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int16_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TFLMRegistration registration, int16_t* output_data);

#ifdef USE_TFLM_COMPRESSION

template <typename TIO, typename CTB>
TfLiteStatus TestConvQuantizedPerChannelCompressed(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* output_dims_data,
    const float* expected_output_data, TIO* expected_output_quantized,
    TIO* output_quantized, float output_scale, int output_zero_point,
    const TfLiteConvParams* conv_params, TFLMRegistration registration,
    const TestCompressionQuantizedInfo<CTB>* comp_info) {
  // TODO(ddavis-2015): account for optional bias tensor
  // bool null_bias = comp_info->bias_data == nullptr ? true : false;

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

  TfLiteAffineQuantization filter_quant = {};
  TfLiteTensor filter_tensor = CreatePerChannelQuantizedTensor(
      comp_info->filter_compressed, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, kConvQuantizedDimension,
      false /* is_variable */, kTfLiteInt8);
  SymmetricPerChannelQuantize(
      comp_info->filter_data, comp_info->filter_value_table,
      ElementCount(*filter_dims), filter_scales->size, filter_scales->data);

  TfLiteAffineQuantization bias_quant = {};
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      comp_info->bias_compressed, bias_dims, input_scale, filter_scales,
      bias_scales, bias_zero_points, &bias_quant, kConvQuantizedDimension,
      false /* is_variable */, typeToTfLiteType<CTB>());
  SymmetricPerChannelQuantize(comp_info->bias_data, comp_info->bias_value_table,
                              ElementCount(*bias_dims), bias_scales->size,
                              bias_scales->data);

  for (int i = 0; i < ElementCount(*bias_dims); i++) {
    int64_t bias_data0 = comp_info->bias_value_table[i];
    MicroPrintf(
        "bias scale %f bias zero_point %d"
        " bias data %f bias data quantized %lld",
        (double)bias_scales->data[i], bias_zero_points->data[i],
        (double)comp_info->bias_data[i], bias_data0);
  }

  constexpr int tensors_size = kMaxTensors;
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
  return ValidateConvGoldens(tensors, tensors_size, expected_output_quantized,
                             output_dims_count, conv_params, registration,
                             output_quantized, 1.0e-5f /* tolerance */,
                             comp_info);
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_H_
