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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_depthwise_conv.h"

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
namespace tflite {
TfLiteStatus DepthwiseConvPrepareHifi(TfLiteContext* context,
                                      TfLiteNode* node) {
  XtensaDepthwiseConvOpData* data =
      static_cast<XtensaDepthwiseConvOpData*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data));

  MicroContext* micro_context = GetMicroContext(context);

  // Calculate scratch memory requirements and request scratch buffer
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const RuntimeShape& output_shape = GetTensorShape(output);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int depth_multiplier = params.depth_multiplier;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int pad_width = data->reference_op_data.padding.width;
  const int pad_height = data->reference_op_data.padding.height;

  int required_scratch = 0;
  // Dilation is currently not supported on HiFi 4 NN Library
  if ((params.dilation_width_factor == 1) &&
      (params.dilation_height_factor == 1)) {
    required_scratch = xa_nn_conv2d_depthwise_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        depth_multiplier, stride_width, stride_height, pad_width, pad_height,
        output_height, output_width, PREC_ASYM8S, 0 /* NHWC */);
    TF_LITE_ENSURE(context, required_scratch > 0);
  }
  TF_LITE_ENSURE_OK(
      context, context->RequestScratchBufferInArena(
                   context, required_scratch, &data->scratch_tensor_index));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus DepthwiseConvEvalHifi(TfLiteContext* context, TfLiteNode* node,
                                   const TfLiteDepthwiseConvParams& params,
                                   const XtensaDepthwiseConvOpData& data,
                                   const TfLiteEvalTensor* input,
                                   const TfLiteEvalTensor* filter,
                                   const TfLiteEvalTensor* bias,
                                   TfLiteEvalTensor* output) {
#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* filter_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kDepthwiseConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kDepthwiseConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  // If dilation is not required use the optimized NN Library kernel.
  // Otherwise call the reference implementation.
  if ((params.dilation_width_factor == 1) &&
      (params.dilation_height_factor == 1) && bias != nullptr) {
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int pad_width = data.reference_op_data.padding.width;
    const int pad_height = data.reference_op_data.padding.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min =
        data.reference_op_data.output_activation_min;
    const int32_t output_activation_max =
        data.reference_op_data.output_activation_max;
    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
#ifdef USE_TFLM_COMPRESSION
    const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(
        micro_context, filter, filter_comp_td,
        data.reference_op_data.weights_scratch_index);
    const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(
        micro_context, bias, bias_comp_td,
        data.reference_op_data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
    const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
    const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
#endif  // USE_TFLM_COMPRESSION
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    int32_t input_data_format = 0;
    int32_t output_data_format = 0;

    uint8_t* p_scratch = static_cast<uint8_t*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (int i = 0; i < batches; i++) {
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
              &output_data[i * output_height * output_width * output_depth],
              filter_data,
              &input_data[i * input_height * input_width * input_depth],
              bias_data, input_height, input_width, input_depth, filter_height,
              filter_width, depth_multiplier, stride_width, stride_height,
              pad_width, pad_height, output_height, output_width,
              -data.reference_op_data.input_zero_point,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift,
              data.reference_op_data.output_zero_point, input_data_format,
              output_data_format, p_scratch),
          0);
    }

    int out_length = batches * output_height * output_width * output_depth;
    TF_LITE_ENSURE_EQ(context,
                      xa_nn_vec_activation_min_max_8_8(
                          output_data, output_data, output_activation_min,
                          output_activation_max, out_length),
                      0);

    return kTfLiteOk;
  }

  reference_integer_ops::DepthwiseConvPerChannel(
      DepthwiseConvParamsQuantized(params, data.reference_op_data),
      data.reference_op_data.per_channel_output_multiplier,
      data.reference_op_data.per_channel_output_shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<int8_t>(
          micro_context, filter, filter_comp_td,
          data.reference_op_data.weights_scratch_index),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetOptionalTensorData<int32_t>(
          micro_context, bias, bias_comp_td,
          data.reference_op_data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(HIFI3) ||defined(HIFI4) || defined(HIFI5)
