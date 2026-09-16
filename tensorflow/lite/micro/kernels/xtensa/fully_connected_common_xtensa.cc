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
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

void* XtensaInitFullyConnected(TfLiteContext* context, const char* buffer,
                               size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if !defined(VISION_P6)
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
#else
  void* data = context->AllocatePersistentBuffer(
      context, sizeof(XtensaFullyConnectedOpData));
#if !defined(HIFIMINI)
  if (InitXtensaContext()) {
    return nullptr;
  }
#endif
  return data;
#endif  // defined(VISION_P6)
}

TfLiteStatus XtensaCalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
  data->is_per_channel = false;

  if (data_type == kTfLiteFloat32) {
    return kTfLiteOk;
  }

  bool is_per_channel = false;
  if (filter->quantization.type == kTfLiteAffineQuantization &&
      filter->quantization.params != nullptr) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    is_per_channel = affine_quantization->scale->size > 1;
  }

  if (is_per_channel) {
    data->is_per_channel = is_per_channel;
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const int per_channel_quantization_size = affine_quantization->scale->size;

    //  Currently only Int8/Int16 are supported for per channel quantization.
    TF_LITE_ENSURE(
        context,
        (input->type == kTfLiteInt8 && filter->type != kTfLiteInt4) ||
            (input->type == kTfLiteInt16 && filter->type != kTfLiteInt4));

    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      per_channel_quantization_size);

    TF_LITE_ENSURE_EQ(
        context, per_channel_quantization_size,
        filter->dims->data[affine_quantization->quantized_dimension]);

    data->per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, per_channel_quantization_size * sizeof(int32_t)));
    data->per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, per_channel_quantization_size * sizeof(int32_t)));

    // Populate multiplier and shift using affine quantization.
    const float input_scale = input->params.scale;
    const float output_scale = output->params.scale;
    const float* filter_scales = affine_quantization->scale->data;

    for (int i = 0; i < per_channel_quantization_size; ++i) {
      const float scale = filter_scales[i];
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t significand;
      int channel_shift;
      QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
      data->per_channel_output_multiplier[i] = significand;
      data->per_channel_output_shift[i] = channel_shift;
    }
  } 
  else {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
#if defined(HIFIMINI)
    if (input->type == kTfLiteInt8) {
      QuantizeMultiplierForInt24(real_multiplier, &data->output_multiplier,
                                 &data->output_shift);
    } else {
      QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                         &data->output_shift);
    }
#else
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
#endif
  }

    // Filter weights will always be symmetric quantized since we only support
    // int8 quantization. See
    // https://github.com/tensorflow/tensorflow/issues/44912 for additional
    // context.
    // Disabling filter ZP check to allow activations * activations FC in Transformer networks.
    //TFLITE_DCHECK(filter->params.zero_point == 0);

    data->input_zero_point = input->params.zero_point;
    data->filter_zero_point = filter->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    return CalculateActivationRangeQuantized(context, activation, output,
                                             &data->output_activation_min,
                                             &data->output_activation_max);
  return kTfLiteOk;
}

TfLiteStatus XtensaPrepareFullyConnected(TfLiteContext* context,
                                         TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  // Disabling this check to support Int8 input, filter and Int16 output variant
  // TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (filter->type == kTfLiteInt4) {
#if defined(HIFI5) && defined(NNLIB_HIFI5)
  const TfLiteEvalTensor* filter_eval =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter_eval);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  const int align_width = 16; 
  int filter_size = align_width + accum_depth;
  context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
#else    
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
#endif                                         
  }

  TFLITE_DCHECK_GE(GetTensorShape(output).DimensionsCount(), 1);

  TF_LITE_ENSURE_OK(context, XtensaCalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  if (micro_context->IsTensorCompressed(node, kFullyConnectedWeightsTensor) &&
      filter->type == kTfLiteInt4) {
    MicroPrintf("Compression not supported with INT4 tensors");
    return kTfLiteError;
  }
  data->weights_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(
          node, kFullyConnectedWeightsTensor);
  data->bias_scratch_index = micro_context->AllocateDecompressionScratchBuffer(
      node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, FullyConnectedPrepareVision(context, node));
#endif  // defined(VISION_P6)

  return kTfLiteOk;
}

}  // namespace tflite
