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

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
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

  // Check if the Xtensa optimized code can be used
  // HIFI4 and HIFI5 do not allow bias data pointer to be nullptr
  /* TODO(b/277112516): Dilation is currently not supported on HiFi 4 NN Library
   */
  bool inputs_and_bias_ok = bias != nullptr;
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  inputs_and_bias_ok =
      inputs_and_bias_ok &&
      (input->type == kTfLiteInt8 ||
       (input->type == kTfLiteInt16 && bias->type == kTfLiteInt64));
#else
  inputs_and_bias_ok = inputs_and_bias_ok && (input->type == kTfLiteInt8);
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  if (!(inputs_and_bias_ok && params->dilation_width_factor == 1 &&
        params->dilation_height_factor == 1 &&
        input_shape.Dims(1) >= filter_shape.Dims(1) &&
        input_shape.Dims(2) >= filter_shape.Dims(2))) {
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
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_channels = output_shape.Dims(3);
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_height = data->reference_op_data.padding.height;
  const int pad_width = data->reference_op_data.padding.width;

  int required_scratch = 0;
  // TODO(b/277112516): Dilation is currently not supported on HiFi 4 NN Library
  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    if (input->type == kTfLiteInt8) {
      if (input_height == 1 && filter_height == 1 && output_height == 1) {
        int inp_h, filt_h, filt_w, str_h, pad_h, out_h;
        inp_h = input_width;
        filt_h = filter_width;
        filt_w = filter_height;
        str_h = stride_width;
        pad_h = pad_width;
        out_h = output_width;
        required_scratch = xa_nn_conv2d_std_getsize(
            inp_h, input_depth, filt_h, filt_w, str_h, pad_h, out_h,
            output_channels, PREC_ASYM8S);
      } else {
        required_scratch = xa_nn_conv2d_std_getsize(
            input_height, input_depth, filter_height, filter_width,
            stride_height, pad_height, output_height, output_channels,
            PREC_ASYM8S);
      }
      TF_LITE_ENSURE(context, required_scratch > 0);
    }
    if (input->type == kTfLiteInt16) {
      if (input_height == 1 && filter_height == 1 && output_height == 1) {
        int inp_h, filt_h, filt_w, str_h, pad_h, out_h;
        inp_h = input_width;
        filt_h = filter_width;
        filt_w = filter_height;
        str_h = stride_width;
        pad_h = pad_width;
        out_h = output_width;
        required_scratch = xa_nn_conv2d_std_getsize(
            inp_h, input_depth, filt_h, filt_w, str_h, pad_h, out_h,
            output_channels, PREC_SYM16S);
      } else {
        required_scratch = xa_nn_conv2d_std_getsize(
            input_height, input_depth, filter_height, filter_width,
            stride_height, pad_height, output_height, output_channels,
            PREC_SYM16S);
      }
      TF_LITE_ENSURE(context, required_scratch > 0);
    }
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

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
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
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
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
  const int64_t* bias_data =
      tflite::micro::GetOptionalTensorData<int64_t>(bias);
#endif  // USE_TFLM_COMPRESSION
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);

  int output_data_format = 0;
  int out_length = output_height * output_width * output_depth;
  if (filter_height == 1 && filter_width == 1) {
    for (int batch = 0; batch < batches; ++batch) {
      int16_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_pointwise_per_chan_sym8sxsym16s(
              p_out_temp, const_cast<WORD8*>(filter_data),
              const_cast<WORD16*>(&input_data[batch * input_height *
                                              input_width * input_depth]),
              const_cast<WORD64*>(bias_data), input_height, input_width,
              input_depth, output_depth, 0,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift, 0,
              output_data_format),
          0);

      TF_LITE_ENSURE_EQ(context,
                        xa_nn_vec_activation_min_max_16_16(
                            p_out_temp, p_out_temp, output_activation_min,
                            output_activation_max, out_length),
                        0);
    }
  } else {
    void* p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));
    for (int batch = 0; batch < batches; ++batch) {
      int16_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      {
        TF_LITE_ENSURE_EQ(
            context,
            xa_nn_conv2d_std_per_chan_sym8sxsym16s(
                p_out_temp,
                &input_data[batch * input_height * input_width * input_depth],
                const_cast<int8_t*>(filter_data),  // filter_data,
                bias_data, input_height, input_width, input_depth,
                filter_height, filter_width, output_depth, stride_width,
                stride_height, pad_width, pad_height, output_height,
                output_width, 0,
                data.reference_op_data.per_channel_output_multiplier,
                data.reference_op_data.per_channel_output_shift, 0,
                output_data_format, static_cast<void*>(p_scratch)),
            0);
      }
      TF_LITE_ENSURE_EQ(context,
                        xa_nn_vec_activation_min_max_16_16(
                            p_out_temp, p_out_temp, output_activation_min,
                            output_activation_max, out_length),
                        0);
    }
  }

  return kTfLiteOk;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

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
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
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
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(
      micro_context, bias, bias_comp_td,
      data.reference_op_data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
  const int32_t* bias_data =
      tflite::micro::GetOptionalTensorData<int32_t>(bias);
#endif  // USE_TFLM_COMPRESSION
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int8_t* filter_data;
  if (filter->type == kTfLiteInt4) {
    int8_t* unpacked_filter_data =
        static_cast<int8_t*>(context->GetScratchBuffer(
            context, data.reference_op_data.filter_buffer_index));
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(filter).FlatSize(), unpacked_filter_data);
    filter_data = unpacked_filter_data;
  } else {
#ifdef USE_TFLM_COMPRESSION
    filter_data = tflite::micro::GetTensorData<int8_t>(
        micro_context, filter, weights_comp_td,
        data.reference_op_data.weights_scratch_index);
#else   // USE_TFLM_COMPRESSION
    filter_data = tflite::micro::GetTensorData<int8_t>(filter);
#endif  // USE_TFLM_COMPRESSION
  }

  int output_data_format = 0;
  int out_length = output_height * output_width * output_depth;

  if (filter_height == 1 && filter_width == 1) {
    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      TF_LITE_ENSURE_EQ(
          context,

          xa_nn_conv2d_pointwise_per_chan_sym8sxasym8s(
              p_out_temp, const_cast<WORD8*>(filter_data),
              const_cast<WORD8*>(&input_data[batch * input_height *
                                             input_width * input_depth]),
              const_cast<WORD32*>(bias_data), input_height, input_width,
              input_depth, output_depth, input_offset,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift, output_offset,
              output_data_format),
          0);

      TF_LITE_ENSURE_EQ(context,
                        xa_nn_vec_activation_min_max_8_8(
                            p_out_temp, p_out_temp, output_activation_min,
                            output_activation_max, out_length),
                        0);
    }
  } else {
    void* p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_out_temp;
      p_out_temp = &output_data[batch * out_length];

      {
        TF_LITE_ENSURE_EQ(
            context,
            xa_nn_conv2d_std_per_chan_sym8sxasym8s(
                p_out_temp,
                &input_data[batch * input_height * input_width * input_depth],
                const_cast<int8_t*>(filter_data),  // filter_data,
                bias_data, input_height, input_width, input_depth,
                filter_height, filter_width, output_depth, stride_width,
                stride_height, pad_width, pad_height, output_height,
                output_width, input_offset,
                data.reference_op_data.per_channel_output_multiplier,
                data.reference_op_data.per_channel_output_shift, output_offset,
                output_data_format, static_cast<void*>(p_scratch)),
            0);
      }

      TF_LITE_ENSURE_EQ(context,
                        xa_nn_vec_activation_min_max_8_8(
                            p_out_temp, p_out_temp, output_activation_min,
                            output_activation_max, out_length),
                        0);
    }
  }

  return kTfLiteOk;
}

}  // namespace tflite
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
