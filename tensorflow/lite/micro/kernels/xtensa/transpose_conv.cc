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

#include "tensorflow/lite/kernels/internal/reference/transpose_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

// For the TfLite transpose_conv implementation, input tensor 0 corresponds to
// the OutputShapeTensor. However, since TFLM does not support dynamic tensors,
// the TFLM implementation ignores input tensor 0 and the only inputs we care
// about are kFilterTensor, kInputTensor and kBiasTensor.
constexpr int kFilterTensor = 1;
constexpr int kInputTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

struct OpData {
  ConvParams params;

  // A scratch buffer is required for quantized implementations.
  int scratch_buffer_index;

#ifdef USE_TFLM_COMPRESSION

  // scratch buffers for compressed tensors
  int filter_scratch_index;
  int bias_scratch_index;

#endif  // USE_TFLM_COMPRESSION

  // TODO(b/192090531): Remove this once all 8x16 transpose conv models use
  // 64-bit biases.
  int bias_converted_buffer_index;

  // Multiplier and shift arrays are required for the int8 implementation.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTransposeConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 4;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int unused_output_width;
  int unused_output_height;
  TfLitePaddingValues padding_values = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1,
      1,  // Dilation height and width are always 1 for transpose_conv.
      height, width, filter_height, filter_width, padding,
      &unused_output_height, &unused_output_width);

  data->params.padding_type = RuntimePaddingType(padding);
  data->params.padding_values.width = padding_values.width;
  data->params.padding_values.height = padding_values.height;

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    MicroContext* micro_context = GetMicroContext(context);
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kInputTensor);
    TF_LITE_ENSURE(context, input != nullptr);
    TfLiteTensor* filter =
        micro_context->AllocateTempInputTensor(node, kFilterTensor);
    TF_LITE_ENSURE(context, filter != nullptr);
    TfLiteTensor* bias =
        micro_context->AllocateTempInputTensor(node, kBiasTensor);
    TfLiteTensor* output =
        micro_context->AllocateTempOutputTensor(node, kOutputTensor);
    TF_LITE_ENSURE(context, output != nullptr);
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, kTfLiteActNone,
        &data->params.output_multiplier, &data->params.output_shift,
        &data->params.quantized_activation_min,
        &data->params.quantized_activation_max,
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        output_channels));

    // TODO(b/192090531): Remove this once all 8x16 transpose conv models use
    // 64-bit biases.
    if (input->type == kTfLiteInt16) {
      TFLITE_DCHECK(filter->type == kTfLiteInt8);
      TFLITE_DCHECK(output->type == kTfLiteInt16);
      if (bias->type == kTfLiteInt16) {
        TFLITE_DCHECK(
            context->RequestScratchBufferInArena(
                context, GetTensorShape(bias).FlatSize() * sizeof(std::int64_t),
                &(data->bias_converted_buffer_index)) == kTfLiteOk);
      }
    }
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    micro_context->DeallocateTempTfLiteTensor(filter);
    if (bias != nullptr) {
      micro_context->DeallocateTempTfLiteTensor(bias);
    }
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteTransposeConvParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kFilterTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  // Get height and width of the output.
  const int width = SizeOfDimension(output, 2);
  const int height = SizeOfDimension(output, 1);
  const int filter_width = SizeOfDimension(filter, 2);
  const int filter_height = SizeOfDimension(filter, 1);

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // Quantized kernels use an int32 scratch buffer.
  if (input->type == kTfLiteInt8) {
    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;

    const int input_height = SizeOfDimension(input, 1);
    const int input_width = SizeOfDimension(input, 2);
    const int input_depth = SizeOfDimension(input, 3);
    const int output_height = height;
    const int output_width = width;
    int32_t scratch_buffer_size = 0;
    scratch_buffer_size = xa_nn_transpose_conv_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        stride_width, stride_height, output_height, output_width, num_channels,
        PREC_SYM8S, PREC_ASYM8S);
    TFLITE_DCHECK(context->RequestScratchBufferInArena(
                      context, scratch_buffer_size,
                      &(data->scratch_buffer_index)) == kTfLiteOk);
#else  // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
    TFLITE_DCHECK(context->RequestScratchBufferInArena(
                      context,
                      GetTensorShape(output).FlatSize() * sizeof(int32_t),
                      &(data->scratch_buffer_index)) == kTfLiteOk);
#endif
  }

  // Quantized 16x8 kernels use an int64 scratch buffer.
  if (input->type == kTfLiteInt16) {
    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;

    const int input_height = SizeOfDimension(input, 1);
    const int input_width = SizeOfDimension(input, 2);
    const int input_depth = SizeOfDimension(input, 3);
    const int output_height = height;
    const int output_width = width;
    int32_t scratch_buffer_size = 0;
    scratch_buffer_size = xa_nn_transpose_conv_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        stride_width, stride_height, output_height, output_width, num_channels,
        PREC_SYM8S, PREC_SYM16S);
    TFLITE_DCHECK(context->RequestScratchBufferInArena(
                      context, scratch_buffer_size,
                      &(data->scratch_buffer_index)) == kTfLiteOk);
#else   // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
    TFLITE_DCHECK(context->RequestScratchBufferInArena(
                      context,
                      GetTensorShape(output).FlatSize() * sizeof(std::int64_t),
                      &(data->scratch_buffer_index)) == kTfLiteOk);
#endif  // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  }

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);

    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height,
                                        input->type, data));

  // Offsets (zero points)
  data->params.input_offset = -input->params.zero_point;
  data->params.weights_offset = -filter->params.zero_point;
  data->params.output_offset = output->params.zero_point;

  // Stride
  data->params.stride_width = params->stride_width;
  data->params.stride_height = params->stride_height;

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  data->filter_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node, kFilterTensor);
  data->bias_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node, kBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 4)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* filter_comp_td =
      micro_context->GetTensorCompressionData(node, kFilterTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      const auto& params =
          *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
      ConvParams op_params = data.params;
      CalculateActivationRange(params.activation,
                               &op_params.float_activation_min,
                               &op_params.float_activation_max);

      reference_ops::TransposeConv(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(
              micro_context, filter, filter_comp_td, data.filter_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(
              micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8: {
      int32_t* scratch_buffer = static_cast<int32_t*>(
          context->GetScratchBuffer(context, data.scratch_buffer_index));
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      if (bias != nullptr && bias->type == kTfLiteInt32) {
        const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
        const RuntimeShape& filter_shape =
            tflite::micro::GetTensorShape(filter);
        const RuntimeShape& output_shape =
            tflite::micro::GetTensorShape(output);
        const int stride_width = data.params.stride_width;
        const int stride_height = data.params.stride_height;
        const int pad_width = data.params.padding_values.width;
        const int pad_height = data.params.padding_values.height;

        const int batches = MatchingDim(input_shape, 0, output_shape, 0);
        const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
        const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

        const int input_height = input_shape.Dims(1);
        const int input_width = input_shape.Dims(2);
        const int filter_height = filter_shape.Dims(1);
        const int filter_width = filter_shape.Dims(2);
        const int output_height = output_shape.Dims(1);
        const int output_width = output_shape.Dims(2);
        const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
#ifdef USE_TFLM_COMPRESSION
        const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(
            micro_context, filter, filter_comp_td, data.filter_scratch_index);
        const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(
            micro_context, bias, bias_comp_td, data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
        const int8_t* filter_data =
            tflite::micro::GetTensorData<int8_t>(filter);
        const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
#endif  // USE_TFLM_COMPRESSION
        int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

        const int num_elements = output_shape.FlatSize();

        for (int b = 0; b < batches; b++) {
          xa_nn_transpose_conv_sym8sxasym8s(
              &output_data[b * output_height * output_width * output_depth],
              const_cast<WORD8*>(
                  &input_data[b * input_height * input_width * input_depth]),
              const_cast<WORD8*>(filter_data), const_cast<WORD32*>(bias_data),
              stride_width, stride_height, pad_width, pad_height, input_depth,
              output_depth, input_height, input_width, filter_height,
              filter_width, output_height, output_width, num_elements / batches,
              data.params.input_offset, data.params.output_offset,
              data.per_channel_output_shift, data.per_channel_output_multiplier,
              scratch_buffer);
        }
      } else {
        reference_integer_ops::TransposeConv(
            data.params, data.per_channel_output_multiplier,
            data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int8_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 filter_comp_td,
                                                 data.filter_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<int32_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output),
            tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
      }
#else
      reference_integer_ops::TransposeConv(
          data.params, data.per_channel_output_multiplier,
          data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(
              micro_context, filter, filter_comp_td, data.filter_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(
              micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
#endif
      break;
    }
    case kTfLiteInt16: {
      std::int64_t* scratch_buffer = static_cast<int64_t*>(
          context->GetScratchBuffer(context, data.scratch_buffer_index));
      // TODO(b/192090531): Remove this once all 8x16 transpose conv models use
      // 64-bit biases.
      if (bias == nullptr || bias->type == kTfLiteInt16) {
        std::int64_t* bias_converted_buffer = nullptr;
        if (bias != nullptr) {
          bias_converted_buffer =
              static_cast<int64_t*>(context->GetScratchBuffer(
                  context, data.bias_converted_buffer_index));
          const int16_t* const bias_int16_data =
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int16_t>(
                  micro_context, bias, bias_comp_td, data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
              static_cast<int16_t*>(bias->data.data);
#endif  // USE_TFLM_COMPRESSION
          for (int i = 0; i < tflite::micro::GetTensorShape(bias).FlatSize();
               i++) {
            bias_converted_buffer[i] = bias_int16_data[i];
          }
        }
        reference_integer_ops::TransposeConv(
            data.params, data.per_channel_output_multiplier,
            data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 filter_comp_td,
                                                 data.filter_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(bias), bias_converted_buffer,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output),
            tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
      } else {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
        const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
        const RuntimeShape& filter_shape =
            tflite::micro::GetTensorShape(filter);
        const RuntimeShape& output_shape =
            tflite::micro::GetTensorShape(output);
        const int stride_width = data.params.stride_width;
        const int stride_height = data.params.stride_height;
        const int pad_width = data.params.padding_values.width;
        const int pad_height = data.params.padding_values.height;

        const int batches = MatchingDim(input_shape, 0, output_shape, 0);
        const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
        const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

        const int input_height = input_shape.Dims(1);
        const int input_width = input_shape.Dims(2);
        const int filter_height = filter_shape.Dims(1);
        const int filter_width = filter_shape.Dims(2);
        const int output_height = output_shape.Dims(1);
        const int output_width = output_shape.Dims(2);
        const int16_t* input_data =
            tflite::micro::GetTensorData<int16_t>(input);
#ifdef USE_TFLM_COMPRESSION
        const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(
            micro_context, filter, filter_comp_td, data.filter_scratch_index);
        const int64_t* bias_data = tflite::micro::GetTensorData<int64_t>(
            micro_context, bias, bias_comp_td, data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
        const int8_t* filter_data =
            tflite::micro::GetTensorData<int8_t>(filter);
        const int64_t* bias_data = tflite::micro::GetTensorData<int64_t>(bias);
#endif  // USE_TFLM_COMPRESSION
        int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);

        const int num_elements = output_shape.FlatSize();

        for (int b = 0; b < batches; b++) {
          xa_nn_transpose_conv_sym8sxsym16s(
              &output_data[b * output_height * output_width * output_depth],
              const_cast<WORD16*>(
                  &input_data[b * input_height * input_width * input_depth]),
              const_cast<WORD8*>(filter_data), const_cast<WORD64*>(bias_data),
              stride_width, stride_height, pad_width, pad_height, input_depth,
              output_depth, input_height, input_width, filter_height,
              filter_width, output_height, output_width, num_elements / batches,
              data.per_channel_output_shift, data.per_channel_output_multiplier,
              scratch_buffer);
        }
#else  // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
        reference_integer_ops::TransposeConv(
            data.params, data.per_channel_output_multiplier,
            data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 filter_comp_td,
                                                 data.filter_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetTensorData<int64_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output),
            tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
#endif  // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_TRANSPOSE_CONV() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
