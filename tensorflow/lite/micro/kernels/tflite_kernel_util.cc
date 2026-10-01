/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <complex>
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/micro/kernels/kernel_util.h"

#ifndef TF_LITE_STATIC_MEMORY
#include <string>
#endif  // TF_LITE_STATIC_MEMORY

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/cppmath.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"

#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif

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

}  // anonymous namespace.

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

#ifndef TF_LITE_STATIC_MEMORY

const TfLiteTensor* GetIntermediates(TfLiteContext* context,
                                     const TfLiteNode* node, int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->intermediates->size, node->intermediates->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

TfLiteStatus GetIntermediatesSafe(const TfLiteContext* context,
                                  const TfLiteNode* node, int index,
                                  TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_STATUS(
      ValidateTensorIndexingSafe(context, index, node->intermediates->size,
                                 node->intermediates->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}
#endif  // TF_LITE_STATIC_MEMORY

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

namespace {

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
    qmin = std::numeric_limits<int8_t>::min();
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

#ifndef TF_LITE_STATIC_MEMORY
// TODO(b/172067338): Having this function be part of TF_LITE_STATIC_MEMORY
// build results in a 6KB size increase, even though the function is unsused for
// that build. What appears to be happening is that while the linker drops the
// unsused function, the string library that gets pulled in is not dropped,
// resulting in the increased binary size.
std::string GetShapeDebugString(const TfLiteIntArray* shape) {
  std::string str;
  for (int d = 0; d < shape->size; ++d) {
    if (str.empty())
      str = "[" + std::to_string(shape->data[d]);
    else
      // Don't add space after "," to make the output consistent with
      // tensorflow::shape_inference::InferenceContext::DebugString()
      str += "," + std::to_string(shape->data[d]);
  }
  if (str.empty()) {
    str = "[]";
  } else {
    str += "]";
  }
  return str;
}

std::string GetTensorDebugString(const TfLiteTensor* tensor) {
  return std::string("{\n  type: ") + TfLiteTypeGetName(tensor->type) +
         "\n  data: {...}\n  dims: " + GetShapeDebugString(tensor->dims) +
         "\n}";
}
#endif  // TF_LITE_STATIC_MEMORY

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

bool IsMobilePlatform() {
#if defined(ANDROID) || defined(__ANDROID__)
  return true;
#elif defined(__APPLE__) && (TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE)
  return true;
#else
  return false;
#endif
}

bool HasUnspecifiedDimension(const TfLiteTensor* tensor) { return false; }

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
