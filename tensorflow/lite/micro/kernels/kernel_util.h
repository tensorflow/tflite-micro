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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"

#ifdef USE_TFLM_COMPRESSION

#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_utils.h"

#endif  // USE_TFLM_COMPRESSION

namespace tflite {
namespace micro {

TFLMRegistration RegisterOp(
    void* (*init)(TfLiteContext* context, const char* buffer, size_t length),
    TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node),
    TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node),
    void (*free)(TfLiteContext* context, void* buffer) = nullptr,
    void (*reset)(TfLiteContext* context, void* buffer) = nullptr);

TFLMInferenceRegistration RegisterOp(
    TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node),
    void (*reset)(TfLiteContext* context, void* buffer) = nullptr);

// Prints out n bytes in a int8_t buffer as hex
void PrintNBytes(const int8_t* tensor_data, int n_bytes,
                 const char* prefix = nullptr);

// Prints out the n bytes in a TfLiteEvalTensor as hex
void PrintNBytes(const TfLiteEvalTensor* tensor, int n_bytes,
                 const char* prefix = nullptr);

// Prints out n bytes in a TfLiteTensor as hex
void PrintNBytes(const TfLiteTensor* tensor, int n_bytes,
                 const char* prefix = nullptr);

// Returns a mutable tensor for a given input index. is_variable must be checked
// during prepare when the full TfLiteTensor is available.
TfLiteEvalTensor* GetMutableEvalInput(const TfLiteContext* context,
                                      const TfLiteNode* node, int index);

// Returns the TfLiteEvalTensor struct for a given input index in a node.
const TfLiteEvalTensor* GetEvalInput(const TfLiteContext* context,
                                     const TfLiteNode* node, int index);

// Returns the TfLiteEvalTensor struct for a given output index in a node.
TfLiteEvalTensor* GetEvalOutput(const TfLiteContext* context,
                                const TfLiteNode* node, int index);

// Returns data for a TfLiteEvalTensor struct that are expected to exist.
template <typename T>
T* GetTensorData(TfLiteEvalTensor* tensor) {
  TFLITE_DCHECK(tensor != nullptr);
  return reinterpret_cast<T*>(tensor->data.raw);
}

// Returns const data for a TfLiteEvalTensor struct that are expected to exist.
template <typename T>
const T* GetTensorData(const TfLiteEvalTensor* tensor) {
  TFLITE_DCHECK(tensor != nullptr);
  return reinterpret_cast<const T*>(tensor->data.raw);
}

// Returns data for a TfLiteEvalTensor struct that could be null.
template <typename T>
T* GetOptionalTensorData(TfLiteEvalTensor* tensor) {
  return tensor == nullptr ? nullptr : reinterpret_cast<T*>(tensor->data.raw);
}

// Returns const data for a TfLiteEvalTensor struct that could be null.
template <typename T>
const T* GetOptionalTensorData(const TfLiteEvalTensor* tensor) {
  return tensor == nullptr ? nullptr
                           : reinterpret_cast<const T*>(tensor->data.raw);
}

#ifdef USE_TFLM_COMPRESSION

// Overloads existing GetOptionalTensorData. If not compressed, this will return
// tensor->data.
template <typename T>
const T* GetOptionalTensorData(MicroContext* micro_context,
                               const TfLiteEvalTensor* tensor,
                               const CompressionTensorData* compression_data,
                               int scratch_buffer_handle) {
  if (tensor == nullptr) {
    return nullptr;
  }
  if (compression_data == nullptr) {
    return reinterpret_cast<const T*>(tensor->data.data);
  }

  void* scratch_buffer = nullptr;
  if (scratch_buffer_handle != -1) {
    scratch_buffer = micro_context->GetScratchBuffer(scratch_buffer_handle);
  } else {
    size_t bytes_to_allocate = EvalTensorBytes(tensor);
    scratch_buffer = micro_context->AllocateDecompressionMemory(
        bytes_to_allocate, MicroArenaBufferAlignment());
  }
  TFLITE_DCHECK(scratch_buffer != nullptr);
  void* uncompressed_data = micro_context->DecompressTensorToBuffer(
      *tensor, *compression_data, scratch_buffer);
  return reinterpret_cast<const T*>(uncompressed_data);
}

// Overloads existing GetTensorData. If not compressed, this will return
// tensor->data.
template <typename T>
const T* GetTensorData(MicroContext* micro_context,
                       const TfLiteEvalTensor* tensor,
                       const CompressionTensorData* compression_data,
                       int scratch_buffer_handle) {
  TFLITE_DCHECK(tensor != nullptr);
  return GetOptionalTensorData<T>(micro_context, tensor, compression_data,
                                  scratch_buffer_handle);
}

#endif  // USE_TFLM_COMPRESSION

// Returns the shape of a TfLiteEvalTensor struct.
const RuntimeShape GetTensorShape(const TfLiteEvalTensor* tensor);

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteEvalTensor* input1,
                    const TfLiteEvalTensor* input2);

PaddingType RuntimePaddingType(TfLitePadding padding);

// Relocate tensor dims from FlatBuffer to the persistent storage arena.
// The old dims data is copied to the new storage area.
// The tensor and eval_tensor must be the same tensor.
// Only use during Prepare phase.
TfLiteStatus CreateWritableTensorDimsWithCopy(TfLiteContext* context,
                                              TfLiteTensor* tensor,
                                              TfLiteEvalTensor* eval_tensor);

// Copy all op input tensors to op output tensors. Requires all op input tensor
// shapes and types to be identical to op output tensor shapes and types.
TfLiteStatus CopyOpInputsToOpOutputs(TfLiteContext* context, TfLiteNode* node);

// Copy all op input tensors to subgraph input tensors. Requires all op input
// tensor shapes and types to be identical to subgraph input tensor shapes and
// types.
TfLiteStatus CopyOpInputsToSubgraphInputs(TfLiteContext* context,
                                          TfLiteNode* node,
                                          MicroGraph* graph_info,
                                          int subgraph_idx,
                                          int first_tensor_idx);

// Copy all op output tensors to subgraph input tensors. Requires all op output
// tensor shapes and types to be identical to subgraph input tensor shapes and
// types.
TfLiteStatus CopyOpOutputsToSubgraphInputs(TfLiteContext* context,
                                           TfLiteNode* node,
                                           MicroGraph* graph_info,
                                           int subgraph_idx);

// Copy all subgraph output tensors to op outputs. Requires all subgraph output
// tensor shapes and types to be identical to op output tensor shapes and types.
TfLiteStatus CopySubgraphOutputsToOpOutputs(TfLiteContext* context,
                                            TfLiteNode* node,
                                            MicroGraph* graph_info,
                                            int subgraph_idx);

// If tensor is INT4, make a new TfLiteEvalTensor with data unpacked into
// a scratch buffer. The returned tensor will have the kTfLiteInt8 type.
// Assume scratch buffer is previously requested in Prepare, and
// scratch_buffer_index can be used to retrieve that buffer.
// If the tensor is not INT4, a shallow copy is returned.
TfLiteEvalTensor MakeUnpackedInt4Tensor(TfLiteContext* context,
                                        int scratch_buffer_index,
                                        const TfLiteEvalTensor* tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetInput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
const TfLiteTensor* GetInput(const TfLiteContext* context,
                             const TfLiteNode* node, int index);

// Same as `GetInput` but returns boolean and uses output argument for tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetInputSafe(context, node, kMyTensorIdx, &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetInputSafe(const TfLiteContext* context, const TfLiteNode* node,
                          int index, const TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetVariableInput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetOutput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetOutput(TfLiteContext* context, const TfLiteNode* node,
                        int index);

// Same as `GetOutput` but returns boolean and uses output argument for tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetOutputSafe(context, node, kMyTensorIdx, &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetOutputSafe(const TfLiteContext* context, const TfLiteNode* node,
                           int index, TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetOptionalInputTensor(context, node, kIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
//
// Deprecated. GetInput has the same functionality.
const TfLiteTensor* GetOptionalInputTensor(const TfLiteContext* context,
                                           const TfLiteNode* node, int index);

inline int NumDimensions(const TfLiteTensor* t) { return t->dims->size; }
inline int SizeOfDimension(const TfLiteTensor* t, int dim) {
  return t->dims->data[dim];
}

inline int NumInputs(const TfLiteNode* node) {
  return node->inputs == nullptr ? 0 : node->inputs->size;
}
inline int NumOutputs(const TfLiteNode* node) {
  return node->outputs == nullptr ? 0 : node->outputs->size;
}

inline int64_t NumElements(const int* dims, int num_dims) {
  int64_t count = 1;
  for (int i = 0; i < num_dims; ++i) {
    count *= dims[i];
  }
  return count;
}

inline int64_t NumElements(const TfLiteIntArray* dims) {
  return NumElements(dims->data, dims->size);
}

inline int64_t NumElements(const TfLiteTensor* t) {
  return NumElements(t->dims);
}

// Determines whether tensor is constant.
// TODO(b/138199592): Introduce new query which checks for constant OR
// persistent-read-only, which would be useful for most tensor kernels that
// are potentially dynamic based on the input tensor value availability at the
// time of prepare.
inline bool IsConstantTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteMmapRo;
}

inline bool IsConstantOrPersistentTensor(const TfLiteTensor* tensor) {
  return IsConstantTensor(tensor) ||
         (tensor->allocation_type == kTfLitePersistentRo);
}

// Determines whether it is a hybrid op - one that has float inputs and
// quantized weights.
inline bool IsHybridOp(const TfLiteTensor* input, const TfLiteTensor* weight) {
  return ((weight->type == kTfLiteUInt8 || weight->type == kTfLiteInt8) &&
          input->type == kTfLiteFloat32);
}

// Check dimensionality match and populate OpData for Conv and DepthwiseConv.
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift);

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels);

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier);

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier);

// Calculates the useful quantized range of an activation layer given its
// activation tensor.
TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max);

// Calculates the useful range of an activation layer given its activation
// tensor.
template <typename T>
void CalculateActivationRange(TfLiteFusedActivation activation,
                              T* activation_min, T* activation_max) {
  if (activation == kTfLiteActRelu) {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  } else if (activation == kTfLiteActRelu6) {
    *activation_min = 0;
    *activation_max = 6;
  } else if (activation == kTfLiteActReluN1To1) {
    *activation_min = -1;
    *activation_max = 1;
  } else {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
}

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2);

// Return the size of given type in bytes. Return 0 in case of string.
int TfLiteTypeGetSize(TfLiteType type);

// Return the size of given type in bits. Returns 0 in case of string.
int TfLiteTypeGetSizeBits(TfLiteType type);

/**
 * Calculates the product of the given dimensions. Returns an error if any of
 * the dimensions is negative or if the product overflows.
 * @param context The context to use for error reporting.
 * @param dims The dimensions to multiply.
 * @param error_message The error message to use if an error is encountered.
 * @param product The output parameter to store the product.
 */
TfLiteStatus CheckedShapeProduct(TfLiteContext* context,
                                 std::initializer_list<int> dims,
                                 const char* error_message, size_t& product);

/**
 * Calculates the product of the given dimensions. Returns an error if any of
 * the dimensions is negative or if the product overflows.
 * @param context The context to use for error reporting.
 * @param dims The dimensions to multiply.
 * @param count The length of the dims array.
 * @param error_message The error message to use if an error is encountered.
 * @param product The output parameter to store the product.
 */
TfLiteStatus CheckedShapeProduct(TfLiteContext* context, const int* dims,
                                 int count, const char* error_message,
                                 size_t& product);

/**
 * Calculates the product of the given dimensions. Returns an error if any of
 * the dimensions is negative or if the product overflows. (Same as above
 * function with dims built on the fly)
 * @param context The context to use for error reporting.
 * @param dims The dimensions to multiply.
 * @param error_message The error message to use if an error is encountered.
 * @param product The output parameter to store the product.
 */
TfLiteStatus CheckedShapeProductToInt(TfLiteContext* context,
                                      std::initializer_list<int> dims,
                                      const char* error_message, int& product);

}  // namespace micro

using micro::CalculateActivationRange;
using micro::CalculateActivationRangeQuantized;
using micro::CheckedShapeProduct;
using micro::CheckedShapeProductToInt;
using micro::GetInput;
using micro::GetInputSafe;
using micro::GetOptionalInputTensor;
using micro::GetOutput;
using micro::GetOutputSafe;
using micro::GetQuantizedConvolutionMultipler;
using micro::GetVariableInput;
using micro::HaveSameShapes;
using micro::IsConstantOrPersistentTensor;
using micro::IsConstantTensor;
using micro::IsHybridOp;
using micro::NumDimensions;
using micro::NumInputs;
using micro::NumOutputs;
using micro::PopulateConvolutionQuantizationParams;
using micro::SizeOfDimension;
using micro::TfLiteTypeGetSize;
using micro::TfLiteTypeGetSizeBits;

#ifndef TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_
using micro::NumElements;
#else
inline int64_t NumElements(const micro::TfLiteIntArray* dims) {
  return micro::NumElements(dims);
}
inline int64_t NumElements(const micro::TfLiteTensor* t) {
  return micro::NumElements(t);
}
#endif  // TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_KERNEL_UTIL_H_
