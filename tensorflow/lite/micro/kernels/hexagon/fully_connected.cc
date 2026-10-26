/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "third_party/hexagon/hexagon_tflm_translation_fully_connected.h"

namespace tflite {
namespace micro {
namespace {

struct HexagonOpDataFullyConnected {
  struct OpDataFullyConnected reference_op_data;
  int32_t* converted_bias;
  int input_u8_scratch_index;
  int output_s32_scratch_index;
  int optimizable;
};

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFusedActivation activation,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(activation, &output_activation_min,
                           &output_activation_max);
  FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  const float* bias_data =
      nullptr != bias ? GetTensorData<float>(bias) : nullptr;

  reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), bias_data, GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               const HexagonOpDataFullyConnected& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  FullyConnectedParams op_params;
  op_params.input_offset = -data.reference_op_data.input_zero_point;
  op_params.weights_offset = -data.reference_op_data.filter_zero_point;
  op_params.output_offset = data.reference_op_data.output_zero_point;
  op_params.output_multiplier = data.reference_op_data.output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  op_params.output_shift = data.reference_op_data.output_shift;
  op_params.quantized_activation_min =
      data.reference_op_data.output_activation_min;
  op_params.quantized_activation_max =
      data.reference_op_data.output_activation_max;

  const int32_t* bias_data =
      nullptr != bias ? GetTensorData<int32_t>(bias) : nullptr;

  reference_integer_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(filter), GetTensorData<int8_t>(filter),
      GetTensorShape(bias), bias_data, GetTensorShape(output),
      GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

void HexagonFullyConnectedOptimizationEvaluation(
    TfLiteContext* context, TfLiteNode* node,
    HexagonOpDataFullyConnected* data) {
  const TfLiteEvalTensor* input =
      GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  TfLiteEvalTensor* output =
      GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  if (input->type == kTfLiteInt8) {
    const RuntimeShape filter_shape = GetTensorShape(filter);
    const RuntimeShape output_shape = GetTensorShape(output);
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
    const int output_depth = output_shape.Dims(1);
    if ((accum_depth % 8 == 0) && (output_depth % 4 == 0) &&
        (data->reference_op_data.filter_zero_point == 0)) {
      data->optimizable = 2;
    } else if ((accum_depth % 8 == 0) && (output_depth % 2 == 0) &&
               (data->reference_op_data.filter_zero_point == 0)) {
      data->optimizable = 1;
    } else {
      data->optimizable = 0;
    }
  } else {
    data->optimizable = 0;
  }
}

TfLiteStatus HexagonFullyConnectedOptimizedPrepare(
    TfLiteContext* context, TfLiteNode* node,
    HexagonOpDataFullyConnected* data) {
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  const RuntimeShape filter_shape = GetTensorShape(filter);
  const RuntimeShape output_shape = GetTensorShape(output);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  int8_t* weights_data = const_cast<int8_t*>(GetTensorData<int8_t>(filter));
  const int32_t* bias_data =
      (bias != nullptr) ? GetTensorData<int32_t>(bias) : nullptr;

  if (bias_data != nullptr) {
    data->converted_bias = const_cast<int32_t*>(bias_data);
  } else {
    data->converted_bias =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, output_depth * sizeof(int32_t)));
    TF_LITE_ENSURE(context, data->converted_bias != nullptr);
  }

  HexagonGenerateBias(data->converted_bias, weights_data, bias_data,
                      data->reference_op_data.input_zero_point + 128,
                      output_depth, 1, accum_depth);
  HexagonInterleaveWeightInplace(weights_data, output_depth, accum_depth, 1);

  TF_LITE_ENSURE_OK(
      context,
      context->RequestScratchBufferInArena(
          context, static_cast<size_t>(batches) * accum_depth * sizeof(uint8_t),
          &data->input_u8_scratch_index));
  TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                 context,
                                 static_cast<size_t>(batches) * output_depth *
                                     sizeof(int32_t),
                                 &data->output_s32_scratch_index));
  return kTfLiteOk;
}

TfLiteStatus HexagonFullyConnectedOptimizedEvalInt8(
    TfLiteContext* context, const HexagonOpDataFullyConnected& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    TfLiteEvalTensor* output) {
  const RuntimeShape filter_shape = GetTensorShape(filter);
  const RuntimeShape output_shape = GetTensorShape(output);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  uint8_t* input_u8_scratch = static_cast<uint8_t*>(
      context->GetScratchBuffer(context, data.input_u8_scratch_index));
  int32_t* output_s32_scratch = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.output_s32_scratch_index));

  const int8_t* input_data = GetTensorData<int8_t>(input);
  const int8_t* filter_data = GetTensorData<int8_t>(filter);
  int8_t* output_data = GetTensorData<int8_t>(output);

  for (size_t i = 0; i < static_cast<size_t>(batches) * accum_depth; ++i) {
    input_u8_scratch[i] = static_cast<uint8_t>(input_data[i] + 128);
  }

  for (int b = 0; b < batches; ++b) {
    const size_t b_accum_offset = static_cast<size_t>(b) * accum_depth;
    const size_t b_output_offset = static_cast<size_t>(b) * output_depth;
    if (data.optimizable == 1) {
      gemm_s32_s8xu8_Nany_Mmod2_Kmod8(
          filter_data, input_u8_scratch + b_accum_offset,
          output_s32_scratch + b_output_offset, output_depth, 1, accum_depth);
    } else {
      gemm_s32_s8xu8_Nany_Mmod4_Kmod8(
          filter_data, input_u8_scratch + b_accum_offset,
          output_s32_scratch + b_output_offset, output_depth, 1, accum_depth);
    }

    for (int c = 0; c < output_depth; ++c) {
      int32_t acc =
          output_s32_scratch[b_output_offset + c] + data.converted_bias[c];
      acc = MultiplyByQuantizedMultiplier(
          acc, data.reference_op_data.output_multiplier,
          data.reference_op_data.output_shift);
      acc += data.reference_op_data.output_zero_point;
      acc = std::max(acc, data.reference_op_data.output_activation_min);
      acc = std::min(acc, data.reference_op_data.output_activation_max);
      output_data[b_output_offset + c] = static_cast<int8_t>(acc);
    }
  }
  return kTfLiteOk;
}

void* HexagonFullyConnectedInit(TfLiteContext* context, const char* buffer,
                                size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(HexagonOpDataFullyConnected));
}

TfLiteStatus HexagonFullyConnectedPrepare(TfLiteContext* context,
                                          TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  HexagonOpDataFullyConnected* data =
      static_cast<HexagonOpDataFullyConnected*>(node->user_data);
  const auto* params =
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

  TF_LITE_ENSURE_OK(
      context, CalculateOpDataFullyConnected(context, params->activation,
                                             input->type, input, filter, bias,
                                             output, &data->reference_op_data));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);

  HexagonFullyConnectedOptimizationEvaluation(context, node, data);

  if (data->optimizable != 0) {
    return HexagonFullyConnectedOptimizedPrepare(context, node, data);
  }
  return kTfLiteOk;
}

TfLiteStatus HexagonFullyConnectedEvalInt8(TfLiteContext* context,
                                           TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const HexagonOpDataFullyConnected& data =
      *(static_cast<const HexagonOpDataFullyConnected*>(node->user_data));

  // This kernel only implements the int8 version of the fully_connected kernel.
  TFLITE_DCHECK(input->type == kTfLiteInt8);
  TFLITE_DCHECK(filter->type == kTfLiteInt8);
  if (bias != nullptr) {
    TFLITE_DCHECK(bias->type == kTfLiteInt32);
  }
  TFLITE_DCHECK(output->type == kTfLiteInt8);

  if (data.optimizable != 0) {
    return HexagonFullyConnectedOptimizedEvalInt8(context, data, input, filter,
                                                  output);
  } else {
    return EvalQuantizedInt8(context, node, data, input, filter, bias, output);
  }
}

TfLiteStatus HexagonFullyConnectedEval(TfLiteContext* context,
                                       TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, params->activation, input, filter, bias,
                       output);

    case kTfLiteInt8:
      return HexagonFullyConnectedEvalInt8(context, node);

    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED() {
  return RegisterOp(HexagonFullyConnectedInit, HexagonFullyConnectedPrepare,
                    HexagonFullyConnectedEval);
}

#if defined(HEXAGON)
TFLMRegistration Register_FULLY_CONNECTED_INT8() {
  return RegisterOp(HexagonFullyConnectedInit, HexagonFullyConnectedPrepare,
                    HexagonFullyConnectedEvalInt8);
}
#endif

const TfLiteEvalTensor*
HexagonLegacyGetEvalInput(const TfLiteContext* context, const TfLiteNode* node, int index) asm(
    "_ZN6tflite5micro12GetEvalInputEPK13TfLiteContextPK10TfLiteNodei");
__attribute__((weak, used)) const TfLiteEvalTensor* HexagonLegacyGetEvalInput(
    const TfLiteContext* context, const TfLiteNode* node, int index) {
  return GetEvalInput(context, node, index);
}

TfLiteEvalTensor*
HexagonLegacyGetEvalOutput(const TfLiteContext* context, const TfLiteNode* node, int index) asm(
    "_ZN6tflite5micro13GetEvalOutputEPK13TfLiteContextPK10TfLiteNodei");
__attribute__((weak, used)) TfLiteEvalTensor* HexagonLegacyGetEvalOutput(
    const TfLiteContext* context, const TfLiteNode* node, int index) {
  return GetEvalOutput(context, node, index);
}

const RuntimeShape
HexagonLegacyGetTensorShape(const TfLiteEvalTensor* tensor) asm(
    "_ZN6tflite5micro14GetTensorShapeEPK16TfLiteEvalTensor");
__attribute__((weak, used)) const RuntimeShape
HexagonLegacyGetTensorShape(const TfLiteEvalTensor* tensor) {
  return GetTensorShape(tensor);
}

}  // namespace micro
}  // namespace tflite
