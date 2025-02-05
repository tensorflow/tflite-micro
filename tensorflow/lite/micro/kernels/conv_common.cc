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
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

const int kConvInputTensor = 0;
const int kConvWeightsTensor = 1;
const int kConvBiasTensor = 2;
const int kConvOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
const int kConvQuantizedDimension = 0;

// Returns a ConvParams struct with all the parameters needed for a
// float computation.
ConvParams ConvParamsFloat(const TfLiteConvParams& params,
                           const OpDataConv& data) {
  ConvParams op_params;
  CalculateActivationRange(params.activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  op_params.padding_type = tflite::micro::RuntimePaddingType(params.padding);
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  return op_params;
}

// Returns a ConvParams struct with all the parameters needed for a
// quantized computation.
ConvParams ConvParamsQuantized(const TfLiteConvParams& params,
                               const OpDataConv& data) {
  ConvParams op_params;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = -data.filter_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier = data.output_multiplier;
  op_params.output_shift = -data.output_shift;
  op_params.padding_type = tflite::micro::RuntimePaddingType(params.padding);
  op_params.padding_values.height = data.padding.height;
  op_params.padding_values.width = data.padding.width;
  op_params.stride_height = params.stride_height;
  op_params.stride_width = params.stride_width;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;
  return op_params;
}

void* ConvInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus CalculateOpDataConv(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteConvParams& params, int width,
                                 int height, int filter_width,
                                 int filter_height, int out_width,
                                 int out_height, const TfLiteType data_type,
                                 OpDataConv* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params.padding;
  data->padding = ComputePaddingHeightWidth(
      params.stride_height, params.stride_width, params.dilation_height_factor,
      params.dilation_width_factor, height, width, filter_height, filter_width,
      padding, &out_height, &out_width);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kConvBiasTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params.activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        output_channels));
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpDataConv* data = static_cast<OpDataConv*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      (input->type == kTfLiteFloat32 && filter->type == kTfLiteFloat32) ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 &&
           (filter->type == kTfLiteInt4 || filter->type == kTfLiteInt8)),
      "Hybrid models are not supported on TFLite Micro.");

  const int input_width = input->dims->data[2];
  const int input_height = input->dims->data[1];
  const int filter_width = filter->dims->data[2];
  const int filter_height = filter->dims->data[1];
  const int output_width = output->dims->data[2];
  const int output_height = output->dims->data[1];

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
    TFLITE_DCHECK(affine_quantization->zero_point != nullptr);

    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, data));

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  if (micro_context->IsTensorCompressed(node, kConvWeightsTensor) &&
      filter->type == kTfLiteInt4) {
    MicroPrintf("Compression not supported with INT4 tensors");
    return kTfLiteError;
  }
  data->weights_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node,
                                                        kConvWeightsTensor);
  data->bias_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}
}  // namespace tflite
