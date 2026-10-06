/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/kernel_util.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/cppmath.h"
#include "tensorflow/lite/micro/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {

namespace {

// Assumes tensor_index is a valid index (in bounds)
inline TfLiteTensor* GetTensorAtIndex(const TfLiteContext* context,
                                      int tensor_index) {
  if (context->tensors != nullptr) {
    return &context->tensors[tensor_index];
  } else {
    return context->GetTensor(context, tensor_index);
  }
}

// Validate in a single place to reduce binary size
inline TfLiteStatus ValidateTensorIndexingSafe(const TfLiteContext* context,
                                               int index, int max_size,
                                               const int* tensor_indices,
                                               int* tensor_index) {
  if (index < 0 || index >= max_size) {
    TF_LITE_KERNEL_LOG(const_cast<TfLiteContext*>(context),
                       "Invalid tensor index %d (not in [0, %d))\n", index,
                       max_size);
    return kTfLiteError;
  }
  if (tensor_indices[index] == kTfLiteOptionalTensor) {
    TF_LITE_KERNEL_LOG(const_cast<TfLiteContext*>(context),
                       "Tensor at index %d was optional but was expected\n",
                       index);
    return kTfLiteError;
  }

  *tensor_index = tensor_indices[index];
  return kTfLiteOk;
}

// Same as above but returns -1 for invalid inputs instead of status + logging
// error.
inline int ValidateTensorIndexing(const TfLiteContext* context, int index,
                                  int max_size, const int* tensor_indices) {
  if (index >= 0 && index < max_size) {
    const int tensor_index = tensor_indices[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

inline TfLiteTensor* GetMutableInput(const TfLiteContext* context,
                                     const TfLiteNode* node, int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->inputs->size, node->inputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

inline TfLiteStatus GetMutableInputSafe(const TfLiteContext* context,
                                        const TfLiteNode* node, int index,
                                        const TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_STATUS(ValidateTensorIndexingSafe(
      context, index, node->inputs->size, node->inputs->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

inline TfLiteStatus Quantize(TfLiteContext* context, float scale,
                             int32_t zero_point, float f, int32_t& q) {
  const float tmp = TfLiteRound(f / scale);
  const bool no_integer_overflow_from_quantization =
      (tmp >= static_cast<float>(std::numeric_limits<int32_t>::min()) &&
       tmp <= static_cast<float>(std::numeric_limits<int32_t>::max()));
  TF_LITE_ENSURE(context, no_integer_overflow_from_quantization);
  q = zero_point + static_cast<int32_t>(tmp);
  return kTfLiteOk;
}

TfLiteStatus CalculateActivationRangeQuantizedImpl(
    TfLiteContext* context, TfLiteFusedActivation activation, int32_t qmin,
    int32_t qmax, TfLiteTensor* output, int32_t* act_min, int32_t* act_max) {
  const auto scale = output->params.scale;
  const auto zero_point = output->params.zero_point;

  int32_t tmp_q;
  if (activation == kTfLiteActRelu) {
    TF_LITE_ENSURE_OK(context,
                      Quantize(context, scale, zero_point, 0.0, tmp_q));
    *act_min = std::max(qmin, tmp_q);
    *act_max = qmax;
  } else if (activation == kTfLiteActRelu6) {
    TF_LITE_ENSURE_OK(context,
                      Quantize(context, scale, zero_point, 0.0, tmp_q));
    *act_min = std::max(qmin, tmp_q);
    TF_LITE_ENSURE_OK(context,
                      Quantize(context, scale, zero_point, 6.0, tmp_q));
    *act_max = std::min(qmax, tmp_q);
  } else if (activation == kTfLiteActReluN1To1) {
    TF_LITE_ENSURE_OK(context,
                      Quantize(context, scale, zero_point, -1.0, tmp_q));
    *act_min = std::max(qmin, tmp_q);
    TF_LITE_ENSURE_OK(context,
                      Quantize(context, scale, zero_point, 1.0, tmp_q));
    *act_max = std::min(qmax, tmp_q);
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterOp(
    void* (*init)(TfLiteContext* context, const char* buffer, size_t length),
    TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node),
    TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node),
    void (*free)(TfLiteContext* context, void* buffer),
    void (*reset)(TfLiteContext* context, void* buffer)) {
  return {/*init=*/init,
          /*free=*/free,
          /*prepare=*/prepare,
          /*invoke=*/invoke,
          /*reset*/ reset,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr};
}

TFLMInferenceRegistration RegisterOp(
    TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node),
    void (*reset)(TfLiteContext* context, void* buffer)) {
  return {
      /*invoke=*/invoke,
      /*reset*/ reset,
  };
}

// Returns a mutable tensor for a given input index. is_variable must be checked
// during prepare when the full TfLiteTensor is available.
TfLiteEvalTensor* GetMutableEvalInput(const TfLiteContext* context,
                                      const TfLiteNode* node, int index) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->inputs->size, node->inputs->data);

  if (tensor_index < 0) {
    return nullptr;
  }

  return context->GetEvalTensor(context, node->inputs->data[index]);
}

// Returns the TfLiteEvalTensor struct for a given input index in a node.
const TfLiteEvalTensor* GetEvalInput(const TfLiteContext* context,
                                     const TfLiteNode* node, int index) {
  return GetMutableEvalInput(context, node, index);
}

// Returns the TfLiteEvalTensor struct for a given output index in a node.
TfLiteEvalTensor* GetEvalOutput(const TfLiteContext* context,
                                const TfLiteNode* node, int index) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  return context->GetEvalTensor(context, node->outputs->data[index]);
}

bool HaveSameShapes(const TfLiteEvalTensor* input1,
                    const TfLiteEvalTensor* input2) {
  TFLITE_DCHECK(input1 != nullptr);
  TFLITE_DCHECK(input2 != nullptr);
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

const RuntimeShape GetTensorShape(const TfLiteEvalTensor* tensor) {
  if (tensor == nullptr || tensor->dims == nullptr) {
    return RuntimeShape();
  }
  TfLiteIntArray* dims = tensor->dims;
  const int dims_size = dims->size;
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims->data);
  return RuntimeShape(dims_size, dims_data);
}

PaddingType RuntimePaddingType(TfLitePadding padding) {
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

// Relocate tensor dims from FlatBuffer to the persistent storage arena.
// The old dims data is copied to the new storage area.
// The tensor and eval_tensor must be the same tensor.
// Only use during Prepare phase.
TfLiteStatus CreateWritableTensorDimsWithCopy(TfLiteContext* context,
                                              TfLiteTensor* tensor,
                                              TfLiteEvalTensor* eval_tensor) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, eval_tensor != nullptr);
  TF_LITE_ENSURE(context, context->AllocatePersistentBuffer != nullptr);
  int ranks = tensor->dims->size;
  size_t alloc_size = TfLiteIntArrayGetSizeInBytes(ranks);
  TfLiteIntArray* new_dims = static_cast<TfLiteIntArray*>(
      context->AllocatePersistentBuffer(context, alloc_size));
  TfLiteIntArray* old_dims = tensor->dims;
  new_dims->size = ranks;
  tensor->dims = new_dims;
  eval_tensor->dims = new_dims;
  for (int i = 0; i < ranks; i++) {
    new_dims->data[i] = old_dims->data[i];
  }

  return kTfLiteOk;
}

// Verify that both tensors have the same type and size, then return the size
// of both tensors in bytes if they are the same, or -1 if they are different.
size_t ValidateAndGetTensorSizes(const TfLiteEvalTensor* tensor1,
                                 const TfLiteEvalTensor* tensor2) {
  TFLITE_DCHECK(tensor1->type == tensor2->type);
  size_t tensor1_size = 0;
  size_t tensor2_size = 0;
  TfLiteEvalTensorByteLength(tensor1, &tensor1_size);
  TfLiteEvalTensorByteLength(tensor2, &tensor2_size);
  return (tensor1_size == tensor2_size) ? tensor1_size : -1;
}

TfLiteStatus CopyOpInputsToOpOutputs(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size == node->outputs->size);
  for (int i = 0; i < node->inputs->size; i++) {
    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, i);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, i);
    int bytes = ValidateAndGetTensorSizes(input, output);
    TF_LITE_ENSURE(context, bytes >= 0);
    // Don't copy if the tensors share memory
    if (output->data.data != input->data.data) {
      memcpy(output->data.data, input->data.data, bytes);
    }
  }
  return kTfLiteOk;
}

//  Args:
//    1. int8_t tensor_data - int8_t buffer of unknown size who's data you'd
//    like
//  to print
//    2. int n_btyes -  a small int representing number of bytes you want to
//    print
//  to debug output. It should always be <= tensor_data's size.
//    3. prefix - optional message you'd like to print before printing bytes
//
//  Purpose:
//    Function takes in parameters above and prints n_bytes bytes from the
//  tensor_data buffer. This can be use to debug  the output of a model and it's
//  op.

void PrintNBytes(const int8_t* tensor_data, int n_bytes, const char* prefix) {
  if (prefix != nullptr) {
    MicroPrintf("%s", prefix);
  }

  for (int i = 0; i < n_bytes; ++i) {
    MicroPrintf(" %x", tensor_data[i]);
  }
  MicroPrintf("\n");
}

// same as the PrintNBytes above but the buffer needs to be extracted out of the
// TfLiteEvalTensor*
void PrintNBytes(const TfLiteEvalTensor* tensor, int n_bytes,
                 const char* prefix) {
  const int8_t* tensor_data = tflite::micro::GetTensorData<int8_t>(tensor);
  PrintNBytes(tensor_data, n_bytes, prefix);
}

// same as the PrintNBytes above but the buffer needs to be extracted out of the
// TfLiteEvalTensor*
void PrintNBytes(const TfLiteTensor* tensor, int n_bytes, const char* prefix) {
  const int8_t* tensor_data = tflite::micro::GetTensorData<int8_t>(tensor);
  PrintNBytes(tensor_data, n_bytes, prefix);
}

TfLiteStatus CopyOpInputsToSubgraphInputs(TfLiteContext* context,
                                          TfLiteNode* node,
                                          MicroGraph* graph_info,
                                          int subgraph_idx,
                                          int first_tensor_idx) {
  TF_LITE_ENSURE(context,
                 static_cast<size_t>(node->inputs->size - first_tensor_idx) ==
                     graph_info->NumSubgraphInputs(subgraph_idx));
  for (int i = 0; i < node->inputs->size - first_tensor_idx; i++) {
    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, i + first_tensor_idx);
    TfLiteEvalTensor* subgraph_input =
        graph_info->GetSubgraphInput(subgraph_idx, i);
    int bytes = ValidateAndGetTensorSizes(input, subgraph_input);
    TF_LITE_ENSURE(context, bytes >= 0);
    // Don't copy if the tensors share memory
    if (subgraph_input->data.data != input->data.data) {
      memcpy(subgraph_input->data.data, input->data.data, bytes);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CopyOpOutputsToSubgraphInputs(TfLiteContext* context,
                                           TfLiteNode* node,
                                           MicroGraph* graph_info,
                                           int subgraph_idx) {
  TF_LITE_ENSURE(context, static_cast<size_t>(node->outputs->size) ==
                              graph_info->NumSubgraphInputs(subgraph_idx));
  for (int i = 0; i < node->outputs->size; i++) {
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, i);
    TfLiteEvalTensor* subgraph_input =
        graph_info->GetSubgraphInput(subgraph_idx, i);
    int bytes = ValidateAndGetTensorSizes(output, subgraph_input);
    TF_LITE_ENSURE(context, bytes >= 0);
    // Don't copy if the tensors share memory
    if (subgraph_input->data.data != output->data.data) {
      memcpy(subgraph_input->data.data, output->data.data, bytes);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CopySubgraphOutputsToOpOutputs(TfLiteContext* context,
                                            TfLiteNode* node,
                                            MicroGraph* graph_info,
                                            int subgraph_idx) {
  if (graph_info->NumSubgraphOutputs(subgraph_idx) == 0) return kTfLiteOk;
  TF_LITE_ENSURE(context, static_cast<size_t>(node->outputs->size) ==
                              graph_info->NumSubgraphOutputs(subgraph_idx));
  for (int i = 0; i < node->outputs->size; i++) {
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, i);
    TfLiteEvalTensor* subgraph_output =
        graph_info->GetSubgraphOutput(subgraph_idx, i);
    int bytes = ValidateAndGetTensorSizes(output, subgraph_output);
    TF_LITE_ENSURE(context, bytes >= 0);
    // Don't copy if the tensors share memory
    if (output->data.data != subgraph_output->data.data) {
      memcpy(output->data.data, subgraph_output->data.data, bytes);
    }
  }
  return kTfLiteOk;
}

TfLiteEvalTensor MakeUnpackedInt4Tensor(TfLiteContext* context,
                                        int scratch_buffer_index,
                                        const TfLiteEvalTensor* tensor) {
  if (tensor->type != kTfLiteInt4) {
    return *tensor;
  }

  TfLiteEvalTensor new_tensor;
  new_tensor.data.data = static_cast<int8_t*>(
      context->GetScratchBuffer(context, scratch_buffer_index));
  new_tensor.dims = tensor->dims;
  new_tensor.type = kTfLiteInt8;
  tflite::micro::tensor_utils::UnpackDenseInt4IntoInt8(
      tflite::micro::GetTensorData<int8_t>(tensor),
      tflite::micro::GetTensorShape(tensor).FlatSize(),
      tflite::micro::GetTensorData<int8_t>(&new_tensor));
  return new_tensor;
}

const TfLiteTensor* GetInput(const TfLiteContext* context,
                             const TfLiteNode* node, int index) {
  return GetMutableInput(context, node, index);
}

TfLiteStatus GetInputSafe(const TfLiteContext* context, const TfLiteNode* node,
                          int index, const TfLiteTensor** tensor) {
  return GetMutableInputSafe(context, node, index, tensor);
}

TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index) {
  TfLiteTensor* tensor = GetMutableInput(context, node, index);
  if (tensor == nullptr) return nullptr;
  return tensor->is_variable ? tensor : nullptr;
}

TfLiteTensor* GetOutput(TfLiteContext* context, const TfLiteNode* node,
                        int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->outputs->size, node->outputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

TfLiteStatus GetOutputSafe(const TfLiteContext* context, const TfLiteNode* node,
                           int index, TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_STATUS(ValidateTensorIndexingSafe(
      context, index, node->outputs->size, node->outputs->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

const TfLiteTensor* GetOptionalInputTensor(const TfLiteContext* context,
                                           const TfLiteNode* node, int index) {
  return GetInput(context, node, index);
}

// Per-axis
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift) {
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  return PopulateConvolutionQuantizationParams(
      context, input, filter, bias, output, activation, multiplier, shift,
      output_activation_min, output_activation_max, per_channel_multiplier,
      per_channel_shift, affine_quantization->scale->size);
}

// Per-axis & per-tensor
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels) {
  TF_LITE_ENSURE_EQ(context, input->quantization.type,
                    kTfLiteAffineQuantization);
  TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                    kTfLiteAffineQuantization);
  // TODO(jianlijianli): Enable bias type check and bias scale == input scale
  // * filter scale for each channel in affine quantization once bias
  // quantization is properly populated.
  // TF_LITE_ENSURE_EQ(context, bias->quantization.type,
  // kTfLiteAffineQuantization);

  // Check data type.
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  const bool is_per_channel = affine_quantization->scale->size > 1;
  if (is_per_channel) {
    //  Currently only Int8/Int16 is supported for per channel quantization.
    TF_LITE_ENSURE(context,
                   input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
    TF_LITE_ENSURE(context,
                   filter->type == kTfLiteInt8 || filter->type == kTfLiteInt4);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, num_channels);
    TF_LITE_ENSURE_EQ(
        context, num_channels,
        filter->dims->data[affine_quantization->quantized_dimension]);
  }

  // Populate multiplier and shift using affine quantization.
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float scale = is_per_channel ? filter_scales[i] : filter_scales[0];
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    tflite::micro::QuantizeMultiplier(effective_output_scale, &significand,
                                      &channel_shift);
    per_channel_multiplier[i] = significand;
    per_channel_shift[i] = channel_shift;
  }

  // Populate scalar quantization parameters.
  // This check on legacy quantization parameters is kept only for backward
  // compatibility.
  if (input->type == kTfLiteUInt8) {
    // Check bias scale == input scale * filter scale.
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;

    // Populate quantization parameters with multiplier and shift.
    tflite::micro::QuantizeMultiplier(real_multiplier, multiplier, &exponent);
    *shift = -exponent;
  }
  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
      input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, activation, output, output_activation_min,
        output_activation_max));
  }
  return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale = static_cast<double>(input->params.scale) *
                                     static_cast<double>(filter->params.scale);
  // The following conditions must be guaranteed by the training pipeline.
  if (bias) {
    const double bias_scale = static_cast<double>(bias->params.scale);
    // Here we're making sure the input_product_scale & bias_scale are about the
    // same. Since we have:
    // (output - output_zp) * output_scale =
    // input_product_scale * input_product + bias * bias_scale ---- (0)
    //
    // (0) equals:
    // (input_product + bias) * input_product_scale ----- (1)
    //           +
    // bias * (bias_scale - input_product_scale)   ------ (2)
    //
    // For the real kernel computation, we're doing (1), so we really need to
    // make sure (2) has minimum impact on the output, so:
    // bias * (bias_scale - input_product_scale) / output_scale should be
    // a small number for an integer.
    // Since normally bias should be within a small range.
    // We should expect (bias_scale - input_product_scale) / output_scale to
    // be a small number like 0.02.
    const double scale_diff = std::abs(input_product_scale - bias_scale);
    const double output_scale = static_cast<double>(output->params.scale);

    TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02);
  }
  return GetQuantizedConvolutionMultipler(context, input, filter, output,
                                          multiplier);
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale =
      static_cast<double>(input->params.scale * filter->params.scale);
  TF_LITE_ENSURE(context, input_product_scale >= 0);
  *multiplier = input_product_scale / static_cast<double>(output->params.scale);

  return kTfLiteOk;
}

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
  if (output->type == kTfLiteUInt8) {
    qmin = std::numeric_limits<uint8_t>::min();
    qmax = std::numeric_limits<uint8_t>::max();
  } else if (output->type == kTfLiteInt8) {
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    qmin = std::numeric_limits<int8_t>::min();
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    qmax = std::numeric_limits<int8_t>::max();
  } else if (output->type == kTfLiteInt16) {
    qmin = std::numeric_limits<int16_t>::min();
    qmax = std::numeric_limits<int16_t>::max();
  } else {
    TF_LITE_ENSURE(context, false);
  }

  return CalculateActivationRangeQuantizedImpl(context, activation, qmin, qmax,
                                               output, act_min, act_max);
}

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

// Size of string is not constant, return 0 in such case.
int TfLiteTypeGetSize(TfLiteType type) {
  int size_bits = TfLiteTypeGetSizeBits(type);
  if (size_bits % 8 == 0) {
    return size_bits / 8;
  } else {
    // For non-byte sized types, return 0.
    return 0;
  }
}

int TfLiteTypeGetSizeBits(TfLiteType type) {
  switch (type) {
    case kTfLiteInt2:
      return 2;
    case kTfLiteInt4:
    case kTfLiteUInt4:
      return 4;
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteFloat8E4M3FN:
    case kTfLiteFloat8E5M2:
      return 8;
    case kTfLiteUInt16:
    case kTfLiteInt16:
    case kTfLiteFloat16:
    case kTfLiteBFloat16:
      return 16;
    case kTfLiteFloat32:
    case kTfLiteInt32:
    case kTfLiteUInt32:
      return 32;
    case kTfLiteInt64:
    case kTfLiteUInt64:
    case kTfLiteFloat64:
    case kTfLiteComplex64:
      return 64;
    case kTfLiteComplex128:
      return 128;
    case kTfLiteBool:
      return sizeof(bool) * 8;
    case kTfLiteString:
    case kTfLiteNoType:
    case kTfLiteResource:
    case kTfLiteVariant:
      break;
  }
  return 0;
}

TfLiteStatus CheckedShapeProduct(TfLiteContext* context,
                                 std::initializer_list<int> dims,
                                 const char* error_message, size_t& product) {
  return CheckedShapeProduct(context, dims.begin(), dims.size(), error_message,
                             product);
}

TfLiteStatus CheckedShapeProduct(TfLiteContext* context, const int* dims,
                                 int count, const char* error_message,
                                 size_t& product) {
  TF_LITE_ENSURE(context, count >= 0);
  TF_LITE_ENSURE(context, dims != nullptr || count == 0);
  size_t checked_count = 1;
  for (int i = 0; i < count; ++i) {
    const int d = dims[i];
    TF_LITE_ENSURE_MSG(context, d >= 0, "Encountered a negative dimension.");
    TF_LITE_ENSURE_MSG(
        context,
        checked_count == 0 ||
            static_cast<size_t>(d) <=
                std::numeric_limits<size_t>::max() / checked_count,
        "%s", error_message);
    checked_count *= d;
  }
  product = checked_count;
  return kTfLiteOk;
}

TfLiteStatus CheckedShapeProductToInt(TfLiteContext* context,
                                      std::initializer_list<int> dims,
                                      const char* error_message, int& product) {
  size_t checked_count = 1;
  for (const int d : dims) {
    TF_LITE_ENSURE_MSG(context, d >= 0, "Encountered a negative dimension.");
    TF_LITE_ENSURE_MSG(
        context,
        checked_count == 0 ||
            static_cast<size_t>(d) <=
                std::numeric_limits<size_t>::max() / checked_count,
        "%s", error_message);
    checked_count *= d;
  }
  TF_LITE_ENSURE_MSG(
      context,
      checked_count <= static_cast<size_t>(std::numeric_limits<int>::max()),
      "%s", error_message);
  product = static_cast<int>(checked_count);
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
