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

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"

namespace tflite {

TfLiteStatus ConvPrepareHifi(TfLiteContext* context, TfLiteNode* node) {
  XtensaConvOpData* data = static_cast<XtensaConvOpData*>(node->user_data);
  const auto params = static_cast<const TfLiteConvParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  // Calculate scratch memory requirements and request scratch buffer
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kConvBiasTensor);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const RuntimeShape& output_shape = GetTensorShape(output);

  bool inputs_and_bias_ok =
      (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 && (!bias || bias->type == kTfLiteInt64)) || 
      input->type == kTfLiteFloat32);

  if (inputs_and_bias_ok == 0) {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(filter);
    micro_context->DeallocateTempTfLiteTensor(output);
    if (bias != nullptr) {
      micro_context->DeallocateTempTfLiteTensor(bias);
    }
    return kTfLiteOk;
  }

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_channels = output_shape.Dims(3);
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_height = data->reference_op_data.padding.height;
  const int pad_width = data->reference_op_data.padding.width;

  int required_scratch = 0;
  // Dilation is currently not supported for kTfLiteInt16 datatype.
  if( ((params->dilation_height_factor > 1) || (params->dilation_width_factor > 1)) 
      && ((input->type == kTfLiteInt8) || (input->type == kTfLiteInt16)) && (input_depth == filter_depth)) {
    // For HiFi5, with nnlib-hifi5 versions 1.7.0 onwards and for HiFi4 with nnlib-hifi4 versions 2.5.0 onwards, 
    // we use the below dilated_conv2d_std getsize() API. For the earlier versions, "output_channels" argument is not needed.
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  if (input->type == kTfLiteInt8) {
    required_scratch = xa_nn_dilated_conv2d_std_getsize(
        input_height, input_depth, filter_height, filter_width, stride_height,
        pad_height, output_height, output_channels, PREC_ASYM8S, params->dilation_height_factor);
  }
  else if (input->type == kTfLiteInt16) {
    required_scratch = xa_nn_dilated_conv2d_std_getsize(
        input_height, input_depth, filter_height, filter_width, stride_height,
        pad_height, output_height, output_channels, PREC_SYM16S, params->dilation_height_factor);
  }
#endif // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
#ifndef HIFI_IQ // Scratchpad may not be required in some cases on HiFi-iQ.
    TF_LITE_ENSURE(context, required_scratch > 0);
#endif
  }
  else if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    if (input->type == kTfLiteInt8) {
      if(input_depth == filter_depth){
        required_scratch = xa_nn_conv2d_std_getsize(
            input_height, input_width, input_depth, filter_height, filter_width, filter_depth, stride_height,
            pad_height, stride_width, pad_width, output_height, output_width, output_channels, PREC_ASYM8S, PREC_SYM8S, params->dilation_height_factor, params->dilation_width_factor, 0/*Out data format*/);
      }
      else{
        required_scratch = xa_nn_conv2d_getsize(
            input_height, input_width, input_depth, filter_height, filter_width, filter_depth, params->dilation_height_factor, params->dilation_width_factor, stride_height,
            pad_height, stride_width, pad_width, output_height, output_width, output_channels, PREC_ASYM8S, PREC_SYM8S, 0/*Out data format*/);        
      }
#ifndef HIFI_IQ // Scratchpad may not be required in some cases on HiFi-iQ.
      TF_LITE_ENSURE(context, required_scratch > 0);
#endif
    }
    if (input->type == kTfLiteInt16) {
      if(input_depth == filter_depth){
        required_scratch = xa_nn_conv2d_std_getsize(
            input_height, input_width, input_depth, filter_height, filter_width, filter_depth, stride_height,
            pad_height, stride_width, pad_width, output_height, output_width, output_channels, PREC_SYM16S, PREC_SYM8S, params->dilation_height_factor, params->dilation_width_factor, 0/*Out data format*/);
      }
      else{
        required_scratch = xa_nn_conv2d_getsize(
            input_height, input_width, input_depth, filter_height, filter_width, filter_depth, params->dilation_height_factor, params->dilation_width_factor, stride_height,
            pad_height, stride_width, pad_width, output_height, output_width, output_channels, PREC_SYM16S, PREC_SYM8S, 0/*Out data format*/);               
      }
#ifndef HIFI_IQ // Scratchpad may not be required in some cases on HiFi-iQ.
      TF_LITE_ENSURE(context, required_scratch > 0);
#endif
    }
#if defined(INCLUDE_FLOAT_OPT)    
     if ((input->type == kTfLiteFloat32) && (input_depth == filter_depth)) {
        required_scratch = xa_nn_conv2d_std_getsize(
            input_height, input_width, input_depth, filter_height, filter_width, filter_depth, stride_height,
            pad_height, stride_width, pad_width, output_height, output_width, output_channels, PREC_F32, PREC_F32, params->dilation_height_factor, params->dilation_width_factor, 0/*Out data format*/);
#ifndef HIFI_IQ // Scratchpad may not be required in some cases on HiFi-iQ.
        TF_LITE_ENSURE(context, required_scratch > 0);
#endif
     }
#endif     
  }
  TF_LITE_ENSURE_OK(
      context, context->RequestScratchBufferInArena(
                   context, required_scratch, &data->scratch_tensor_index));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(output);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  return kTfLiteOk;
}

#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)
TfLiteStatus ConvPrepareHifiInt4(TfLiteContext* context, TfLiteNode* node) {
  XtensaConvOpData* data = static_cast<XtensaConvOpData*>(node->user_data);
  const auto params = static_cast<const TfLiteConvParams*>(node->builtin_data);

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

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const RuntimeShape& output_shape = GetTensorShape(output);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_channels = output_shape.Dims(3);
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_height = data->reference_op_data.padding.height;
  const int pad_width = data->reference_op_data.padding.width;

  int required_scratch = 0;

  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1) && 
      (filter_depth == input_depth)) {
        required_scratch = xa_nn_conv2d_std_getsize(
            input_height, input_width, input_depth, filter_height, filter_width, filter_depth, stride_height,
            pad_height, stride_width, pad_width, output_height, output_width, output_channels, PREC_ASYM8S, PREC_SYM4S, params->dilation_height_factor, params->dilation_width_factor, 0/*Out data format*/);
        TF_LITE_ENSURE(context, required_scratch > 0);
  }
  else
  {
    required_scratch =
        RuntimeShape(filter->dims->size,
                    reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();      
  }
  TF_LITE_ENSURE_OK(
      context, context->RequestScratchBufferInArena(
                   context, required_scratch, &data->scratch_tensor_index));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}
#endif

TfLiteStatus ConvEvalHifiInt16(TfLiteContext* context, TfLiteNode* node,
                               const TfLiteConvParams& params,
                               const XtensaConvOpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = data.reference_op_data.padding.width;
  const int pad_height = data.reference_op_data.padding.height;
  const int32_t output_activation_min =
      data.reference_op_data.output_activation_min;
  const int32_t output_activation_max =
      data.reference_op_data.output_activation_max;

  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  const int16_t* input_data = tflite::micro::GetTensorData<int16_t>(input);
#ifdef USE_TFLM_COMPRESSION
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(
      micro_context, filter, weights_comp_td,
      data.reference_op_data.weights_scratch_index);
  const int64_t* bias_data = tflite::micro::GetOptionalTensorData<int64_t>(
      micro_context, bias, bias_comp_td,
      data.reference_op_data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int64_t* bias_data = tflite::micro::GetOptionalTensorData<int64_t>(bias);
#endif  // USE_TFLM_COMPRESSION
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);

  int output_data_format = 0;
  int out_length = output_height * output_width * output_depth;
  if(params.dilation_height_factor == 1 && params.dilation_width_factor == 1) {
    if (filter_height == 1 && filter_width == 1) {
      for (int batch = 0; batch < batches; ++batch) {
        int16_t* p_out_temp;
        p_out_temp = &output_data[batch * out_length];

        TF_LITE_ENSURE_EQ(
            context,
            xa_nn_conv2d_pointwise_v2_per_chan_sym8sxsym16s(
                p_out_temp, const_cast<WORD8*>(filter_data),
                const_cast<WORD16*>(&input_data[batch * input_height *
                                                input_width * input_depth]),
                const_cast<WORD64*>(bias_data), input_height, input_width,
                input_depth, output_depth, 0,
                data.reference_op_data.per_channel_output_multiplier,
                data.reference_op_data.per_channel_output_shift, 0,
                output_data_format, output_activation_min, output_activation_max, NULL),
          0);
      }
    } else {
      void* p_scratch = static_cast<void*>(
          context->GetScratchBuffer(context, data.scratch_tensor_index));

      for (int batch = 0; batch < batches; ++batch) {
        int16_t* p_out_temp;
        p_out_temp = &output_data[batch * out_length];

        {
          if(filter_depth == input_depth){
          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_conv2d_std_v2_per_chan_sym8sxsym16s(
                  p_out_temp,
                  &input_data[batch * input_height * input_width * input_depth],
                  const_cast<int8_t*>(filter_data),  // filter_data,
                  bias_data, input_height, input_width, input_depth,
                  filter_height, filter_width, output_depth, stride_width,
                  stride_height, pad_width, pad_height, output_height,
                  output_width, 0,
                  data.reference_op_data.per_channel_output_multiplier,
                  data.reference_op_data.per_channel_output_shift, 0,
                  output_data_format, static_cast<void*>(p_scratch),
                  output_activation_min, output_activation_max, NULL),
              0);
          }
          else{
            TF_LITE_ENSURE_EQ(
              context,
                xa_nn_conv2d_v2_per_chan_sym8sxsym16s(
                    p_out_temp,
                    &input_data[batch * input_height * input_width * input_depth],
                    const_cast<int8_t*>(filter_data),  // filter_data,
                    bias_data, input_height, input_width, input_depth,
                    filter_height, filter_width, filter_depth, params.dilation_height_factor, params.dilation_width_factor, output_depth, stride_width,
                    stride_height, pad_width, pad_height, output_height,
                    output_width, 0,
                    data.reference_op_data.per_channel_output_multiplier,
                    data.reference_op_data.per_channel_output_shift,
                    0, output_data_format,
                    static_cast<void*>(p_scratch),
                    output_activation_min, output_activation_max, NULL),
                0);
          }
        }
      }
    }
  } else if (filter_depth == input_depth) {
   /* dilated convolution available only for filter_depth = input_depth */
    void* p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (int batch = 0; batch < batches; ++batch) {
      int16_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_dilated_conv2d_std_v2_per_chan_sym8sxsym16s(
              p_out_temp,
              &input_data[batch * input_height * input_width * input_depth],
              const_cast<int8_t*>(filter_data),  // filter_data,
              bias_data, input_height, input_width, input_depth,
              filter_height, filter_width, output_depth, stride_width,
              stride_height, pad_width, pad_height, output_height,
              output_width, 0,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift, 0,
              output_data_format, static_cast<void*>(p_scratch),
              params.dilation_height_factor, params.dilation_width_factor,
              output_activation_min, output_activation_max, NULL),
          0);
    }
  } else {
    reference_integer_ops::ConvPerChannel(
        ConvParamsQuantized(params, data.reference_op_data),
        data.reference_op_data.per_channel_output_multiplier,
        data.reference_op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  }

  return kTfLiteOk;
}

TfLiteStatus ConvEvalHifiInt8(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteConvParams& params,
                              const XtensaConvOpData& data,
                              const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* filter,
                              const TfLiteEvalTensor* bias,
                              TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int32_t input_offset = -data.reference_op_data.input_zero_point;
  const int32_t output_offset = data.reference_op_data.output_zero_point;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = data.reference_op_data.padding.width;
  const int pad_height = data.reference_op_data.padding.height;
  const int32_t output_activation_min =
      data.reference_op_data.output_activation_min;
  const int32_t output_activation_max =
      data.reference_op_data.output_activation_max;

  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);  
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
#ifdef USE_TFLM_COMPRESSION
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(
      micro_context, filter, weights_comp_td,
      data.reference_op_data.weights_scratch_index);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(
      micro_context, bias, bias_comp_td,
      data.reference_op_data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
#endif  // USE_TFLM_COMPRESSION
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  int output_data_format = 0;
  int out_length = output_height * output_width * output_depth;

  if (filter_height == 1 && filter_width == 1 && stride_width == 1 && stride_height == 1 && 
      pad_width == 0 && pad_height == 0 && (input_height == output_height) && (input_width == output_width) && (filter_depth == input_depth)) {
    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_pointwise_v2_per_chan_sym8sxasym8s(
              p_out_temp, const_cast<WORD8*>(filter_data),
              const_cast<WORD8*>(&input_data[batch * input_height *
                                             input_width * input_depth]),
              const_cast<WORD32*>(bias_data), input_height, input_width,
              input_depth, output_depth, input_offset,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift, output_offset,
              output_data_format,
              output_activation_min, output_activation_max, NULL),
          0);
    }
  } else {
    void* p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    if(((params.dilation_width_factor > 1)  || (params.dilation_height_factor > 1)) && (filter_depth != input_depth) )
    {
      /*Dilated Group-conv not supported*/
      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(params, data.reference_op_data),
          data.reference_op_data.per_channel_output_multiplier,
          data.reference_op_data.per_channel_output_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
          return kTfLiteOk;
    }

    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      if (((params.dilation_width_factor > 1)  ||
          (params.dilation_height_factor > 1)) && 
          (filter_depth == input_depth))
      {
        TF_LITE_ENSURE_EQ(
            context,
            xa_nn_dilated_conv2d_std_v2_per_chan_sym8sxasym8s(p_out_temp,
              &input_data[batch * input_height * input_width * input_depth],
              const_cast<int8_t*>(filter_data),  // filter_data,
              bias_data, input_height, input_width, input_depth, filter_height,
              filter_width, output_depth, stride_width, stride_height, pad_width,
              pad_height, output_height, output_width, input_offset,
              data.reference_op_data.per_channel_output_multiplier, data.reference_op_data.per_channel_output_shift,
              output_offset, output_data_format,
              static_cast<void*>(p_scratch), params.dilation_height_factor, params.dilation_width_factor,
              output_activation_min, output_activation_max, NULL),
            0);
      }
      else
      {
        if(filter_depth == input_depth){
          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_conv2d_std_v2_per_chan_sym8sxasym8s(
                  p_out_temp,
                  &input_data[batch * input_height * input_width * input_depth],
                  const_cast<int8_t*>(filter_data),
                  bias_data, input_height, input_width, input_depth,
                  filter_height, filter_width, output_depth, stride_width,
                  stride_height, pad_width, pad_height, output_height,
                  output_width, input_offset,
                  data.reference_op_data.per_channel_output_multiplier,
                  data.reference_op_data.per_channel_output_shift,
                  output_offset, output_data_format,
                  static_cast<void*>(p_scratch),
                  output_activation_min, output_activation_max, NULL),
              0);
        }
        else{
          TF_LITE_ENSURE_EQ(
            context,
              xa_nn_conv2d_v2_per_chan_sym8sxasym8s(
                  p_out_temp,
                  &input_data[batch * input_height * input_width * input_depth],
                  const_cast<int8_t*>(filter_data),  // filter_data,
                  bias_data, input_height, input_width, input_depth,
                  filter_height, filter_width, filter_depth, params.dilation_height_factor, params.dilation_width_factor, output_depth, stride_width,
                  stride_height, pad_width, pad_height, output_height,
                  output_width, input_offset,
                  data.reference_op_data.per_channel_output_multiplier,
                  data.reference_op_data.per_channel_output_shift,
                  output_offset, output_data_format,
                  static_cast<void*>(p_scratch),
                  output_activation_min, output_activation_max, NULL),
              0);              
        }
      }
    }
  }

  return kTfLiteOk;
}

#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)
TfLiteStatus ConvEvalHifiInt4(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteConvParams& params,
                              const XtensaConvOpData& data,
                              const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* filter,
                              const TfLiteEvalTensor* bias,
                              TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);

  const int32_t input_offset = -data.reference_op_data.input_zero_point;
  const int32_t output_offset = data.reference_op_data.output_zero_point;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = data.reference_op_data.padding.width;
  const int pad_height = data.reference_op_data.padding.height;
  const int32_t output_activation_min =
      data.reference_op_data.output_activation_min;
  const int32_t output_activation_max =
      data.reference_op_data.output_activation_max;

  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
#ifdef USE_TFLM_COMPRESSION
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(
      micro_context, filter, weights_comp_td,
      data.reference_op_data.weights_scratch_index);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(
      micro_context, bias, bias_comp_td,
      data.reference_op_data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
#endif  // USE_TFLM_COMPRESSION
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  int output_data_format = 0;
  int out_length = output_height * output_width * output_depth;

  if ((params.dilation_width_factor == 1)  &&
      (params.dilation_height_factor == 1) &&
      (filter_depth == input_depth))
  {
    void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

#ifndef HIFI_IQ
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_std_per_chan_sym4sxasym8s(
            p_out_temp,
              &input_data[batch * input_height * input_width * input_depth],
              filter_data, bias_data, 
              input_height, input_width, input_depth,
              filter_height, filter_width, output_depth,
              stride_width, stride_height, pad_width, pad_height,
              output_height, output_width, input_offset,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift,
              output_offset, output_data_format,
              static_cast<void*>(p_scratch)),
          0);

      TF_LITE_ENSURE_EQ(context,
                        xa_nn_vec_activation_min_max_8_8(
                            p_out_temp, p_out_temp, output_activation_min,
                            output_activation_max, out_length),
                      0);
#else
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_std_v2_per_chan_sym4sxasym8s(
            p_out_temp,
              &input_data[batch * input_height * input_width * input_depth],
              filter_data,bias_data, 
              input_height, input_width, input_depth,
              filter_height, filter_width, output_depth,
              stride_width, stride_height, pad_width, pad_height,
              output_height, output_width, input_offset,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift,
              output_offset, output_data_format,
              static_cast<void*>(p_scratch),
              output_activation_min, output_activation_max, NULL),
          0);
#endif
      }
    }
    else
    {
      int8_t* unpacked_filter_data = static_cast<int8_t*>(
          context->GetScratchBuffer(context, data.scratch_tensor_index));

      tflite::tensor_utils::UnpackDenseInt4IntoInt8(
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(filter).FlatSize(), unpacked_filter_data);
      filter_data = unpacked_filter_data;   

      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(params, data.reference_op_data),
          data.reference_op_data.per_channel_output_multiplier,
          data.reference_op_data.per_channel_output_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          filter_data,
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));        
    }
    return kTfLiteOk;
}
#endif // #if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)

#if defined(INCLUDE_FLOAT_OPT)

TfLiteStatus ConvEvalHifiFloat32(TfLiteContext* context, TfLiteNode* node,
                               const TfLiteConvParams& params,
                               const XtensaConvOpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = data.reference_op_data.padding.width;
  const int pad_height = data.reference_op_data.padding.height;

  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));  
  ConvParams op_params = ConvParamsFloat(params, op_data.reference_op_data);
  
  const float32_t* input_data = tflite::micro::GetTensorData<float32_t>(input);
#ifdef USE_TFLM_COMPRESSION
  const float32_t* filter_data = tflite::micro::GetTensorData<float32_t>(
      micro_context, filter, weights_comp_td,
      data.reference_op_data.weights_scratch_index);
  const float32_t* bias_data = tflite::micro::GetOptionalTensorData<float32_t>(
      micro_context, bias, bias_comp_td,
      data.reference_op_data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
  const float32_t* filter_data = tflite::micro::GetTensorData<float32_t>(filter);
  const float32_t* bias_data = tflite::micro::GetOptionalTensorData<float32_t>(bias);
#endif  // USE_TFLM_COMPRESSION
  float32_t* output_data = tflite::micro::GetTensorData<float32_t>(output);

  int output_data_format = 0;
  int out_length = output_height * output_width * output_depth;
  int err;
  if (filter_height == 1 && filter_width == 1) {
    for (int batch = 0; batch < batches; ++batch) {
      float32_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_pointwise_f32(
              p_out_temp, const_cast<float32_t*>(filter_data),
              const_cast<float32_t*>(&input_data[batch * input_height *
                                              input_width * input_depth]),
              const_cast<float32_t*>(bias_data), input_height, input_width,
              input_depth, output_depth, output_data_format),
          0);

      err = xa_nn_vec_activation_min_max_f32_f32(
          p_out_temp,
          p_out_temp,
          op_params.float_activation_min,
          op_params.float_activation_max,
          out_length);
      TF_LITE_ENSURE(context, err == 0);
    }
  } else if ((filter_depth == input_depth) &&
           ((params.dilation_width_factor == 1) &&
            (params.dilation_height_factor == 1)))
  {
    void* p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (int batch = 0; batch < batches; ++batch) {
      float32_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_std_f32(
              p_out_temp,
              &input_data[batch * input_height * input_width * input_depth],
              const_cast<float32_t*>(filter_data),  // filter_data,
              bias_data, input_height, input_width, input_depth,
              filter_height, filter_width, output_depth, stride_width,
              stride_height, pad_width, pad_height, output_height,
              output_width,output_data_format, static_cast<void*>(p_scratch)),
          0);

      err = xa_nn_vec_activation_min_max_f32_f32(
          p_out_temp,
          p_out_temp,
          op_params.float_activation_min,
          op_params.float_activation_max,
          out_length);
      TF_LITE_ENSURE(context, err == 0);
    }
  }
  else{
    TFLITE_DCHECK(node->user_data != nullptr);
    const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));  
    tflite::reference_ops::Conv(
        ConvParamsFloat(params, op_data.reference_op_data),
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<float>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<float>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetOptionalTensorData<float>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output),
        tflite::micro::GetTensorShape(nullptr), nullptr);        
  }

  return kTfLiteOk;
}
#endif

}  // namespace tflite
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
